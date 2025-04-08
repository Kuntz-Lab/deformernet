import torch
from torch.nn import Module

from .common import *
from .encoders import *
from .diffusion_goal_idx import *
from .flow import *

class BaoFlowVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        self.flow = build_latent_flow(args)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )
        # self.task_ctx_encoder = nn.Sequential(
        #     nn.Linear(1, args.latent_dim),
        #     nn.ReLU()
        # )
        
        self.task_ctx_encoder = TaskContextEncoderPointNet(args.latent_dim)
        
        print("\n BLEO goal idx\n")
        

    def get_loss(self, x, context_pc, init_pc, goal_idx, kl_weight, writer=None, it=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        # print(x.size())
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        # print(z.shape)
        
        # H[Q(z|X)]
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )

        # P(z), Prior probability, parameterized by the flow: z -> w.
        w, delta_log_pw = self.flow(z, torch.zeros([batch_size, 1]).to(z), reverse=False)
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)   # (B, 1)
        log_pz = log_pw - delta_log_pw.view(batch_size, 1)  # (B, 1)

        # Negative ELBO of P(X|z)
        # neg_elbo = self.diffusion.get_loss(x, z)
        # print("task_ctx.shape, z.shape:", task_ctx.shape, z.shape)
        # task_ctx = self.task_ctx_encoder(task_ctx)
        # task_ctx = self.task_ctx_encoder(context_pc, init_pc)
        task_ctx = self.task_ctx_encoder(context_pc, init_pc, goal_idx)
        # print("task_ctx.shape:", task_ctx.shape)
        neg_elbo = self.diffusion.get_loss(x, torch.cat([z, task_ctx], dim=1))
        # neg_elbo = self.diffusion.get_loss(x, torch.cat([z, task_ctx, goal_idx], dim=1))       
        
        
        # Loss
        loss_entropy = -entropy.mean()
        loss_prior = -log_pz.mean()
        loss_recons = neg_elbo
        loss = kl_weight*(loss_entropy + loss_prior) + neg_elbo

        if writer is not None:
            writer.add_scalar('train/loss_entropy', loss_entropy, it)
            writer.add_scalar('train/loss_prior', loss_prior, it)
            writer.add_scalar('train/loss_recons', loss_recons, it)
            writer.add_scalar('train/z_mean', z_mu.mean(), it)
            writer.add_scalar('train/z_mag', z_mu.abs().max(), it)
            writer.add_scalar('train/z_var', (0.5*z_sigma).exp().mean(), it)

        return loss

    def sample(self, w, context_pc, init_pc, num_points, flexibility, truncate_std=None):
        goal_idx = np.random.randint(0, 2, size=(1,))
        goal_idx = torch.from_numpy(goal_idx).unsqueeze(0).repeat(context_pc.shape[0], 256).float().to(context_pc.device)

        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)
        
        # Concat task context with z
        task_ctx = self.task_ctx_encoder(context_pc, init_pc, goal_idx)
        
        # z = torch.cat([z, task_ctx], dim=1)
        z = torch.cat([z, task_ctx], dim=1)
        
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility)
        return samples
 

class TaskContextEncoderPointNet(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        
        # For contextual point cloud
        self.conv1_context = nn.Conv1d(input_dim, 128, 1)
        self.conv2_context = nn.Conv1d(128, 128, 1)
        self.conv3_context = nn.Conv1d(128, 256, 1)

        self.bn1_context = nn.BatchNorm1d(128)
        self.bn2_context = nn.BatchNorm1d(128)
        self.bn3_context = nn.BatchNorm1d(256)

        # For initial point cloud
        self.conv1_init = nn.Conv1d(input_dim, 128, 1)
        self.conv2_init = nn.Conv1d(128, 128, 1)
        self.conv3_init = nn.Conv1d(128, 256, 1)

        self.bn1_init = nn.BatchNorm1d(128)
        self.bn2_init = nn.BatchNorm1d(128)
        self.bn3_init = nn.BatchNorm1d(256)

        # For goal index
        self.fc1_goal_idx = nn.Linear(256, 256)
        self.fc_bn1_goal_idx = nn.BatchNorm1d(256)

        # Fully connected layers
        self.fc1 = nn.Linear(256*3, zdim)
        self.fc_bn1 = nn.BatchNorm1d(zdim)


    def forward(self, context_pc, init_pc, goal_idx):
        x = context_pc.transpose(1, 2)
        x = F.relu(self.bn1_context(self.conv1_context(x)))
        x = F.relu(self.bn2_context(self.conv2_context(x)))
        x = self.bn3_context(self.conv3_context(x))
        x = torch.max(x, 2, keepdim=True)[0]
        context_pc_feat = x.view(-1, self.zdim)

        x = init_pc.transpose(1, 2)
        x = F.relu(self.bn1_init(self.conv1_init(x)))
        x = F.relu(self.bn2_init(self.conv2_init(x)))
        x = self.bn3_init(self.conv3_init(x))
        x = torch.max(x, 2, keepdim=True)[0]
        init_pc_feat = x.view(-1, self.zdim)

        goal_idx_feat = F.relu(self.fc_bn1_goal_idx(self.fc1_goal_idx(goal_idx)))
        
        x = torch.cat([context_pc_feat, init_pc_feat, goal_idx_feat], dim=1)
        x = F.relu(self.fc_bn1(self.fc1(x)))

        return x


# class TaskContextEncoderPointNet(nn.Module):
#     def __init__(self, zdim, input_dim=3):
#         super().__init__()
#         self.zdim = zdim
        
#         # For contextual point cloud
#         self.conv1_context = nn.Conv1d(input_dim, 128, 1)
#         self.conv2_context = nn.Conv1d(128, 128, 1)
#         self.conv3_context = nn.Conv1d(128, 256, 1)

#         self.bn1_context = nn.BatchNorm1d(128)
#         self.bn2_context = nn.BatchNorm1d(128)
#         self.bn3_context = nn.BatchNorm1d(256)

#         # For initial point cloud
#         self.conv1_init = nn.Conv1d(input_dim, 128, 1)
#         self.conv2_init = nn.Conv1d(128, 128, 1)
#         self.conv3_init = nn.Conv1d(128, 256, 1)

#         self.bn1_init = nn.BatchNorm1d(128)
#         self.bn2_init = nn.BatchNorm1d(128)
#         self.bn3_init = nn.BatchNorm1d(256)

#         # Fully connected layers
#         self.fc1 = nn.Linear(512, zdim)
#         self.fc_bn1 = nn.BatchNorm1d(zdim)


#     def forward(self, context_pc, init_pc):
#         x = context_pc.transpose(1, 2)
#         x = F.relu(self.bn1_context(self.conv1_context(x)))
#         x = F.relu(self.bn2_context(self.conv2_context(x)))
#         x = self.bn3_context(self.conv3_context(x))
#         x = torch.max(x, 2, keepdim=True)[0]
#         context_pc_feat = x.view(-1, self.zdim)

#         x = init_pc.transpose(1, 2)
#         x = F.relu(self.bn1_init(self.conv1_init(x)))
#         x = F.relu(self.bn2_init(self.conv2_init(x)))
#         x = self.bn3_init(self.conv3_init(x))
#         x = torch.max(x, 2, keepdim=True)[0]
#         init_pc_feat = x.view(-1, self.zdim)
        
#         x = torch.cat([context_pc_feat, init_pc_feat], dim=1)
#         x = F.relu(self.fc_bn1(self.fc1(x)))

#         return x
