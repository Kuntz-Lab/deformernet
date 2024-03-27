import numpy as np
import matplotlib.pyplot as plt

# def get_y_to_x_ratio(start_point, end_point):
#     theta = np.arctan((end_point[1] - start_point[1]) / (end_point[0] - start_point[0]))
#     print("****theta:", theta)
#     return [-np.tan(np.pi/4 - theta), np.tan(np.pi/4 + theta)]
#     # return [-np.tan(np.pi/2 - theta), np.tan(np.pi/2 + theta)]

def get_eef_position(s, e, alpha, magnitude, vis=False):
    s = np.array(s, dtype=float)
    e = np.array(e, dtype=float)
    alpha_rad = np.radians(alpha)
    midpoint = (s + e) / 2

    # Direction vector from s to e and normalize it
    direction_se = e - s
    normalized_se = direction_se / np.linalg.norm(direction_se)
    # Perpendicular direction for "above"
    perpendicular_se = np.array([-normalized_se[1], normalized_se[0]])

    # Rotate SM to get A
    direction_sm = midpoint - s
    rotation_matrix_alpha = np.array([
        [np.cos(alpha_rad), -np.sin(alpha_rad)], 
        [np.sin(alpha_rad), np.cos(alpha_rad)]
    ])
    a = midpoint + magnitude * np.dot(rotation_matrix_alpha, direction_sm / np.linalg.norm(direction_sm))

    # Do the same thing for B
    alpha_rad = np.radians(180-alpha)
    direction_sm = midpoint - s
    rotation_matrix_alpha = np.array([
        [np.cos(alpha_rad), -np.sin(alpha_rad)], 
        [np.sin(alpha_rad), np.cos(alpha_rad)]
    ])
    b = midpoint + magnitude * np.dot(rotation_matrix_alpha, direction_sm / np.linalg.norm(direction_sm))

    if vis:
        visualize_result(s, e, a, b, midpoint)
    
    return a, b, midpoint


def calculate_angle(p1, p2, p3):
    vector_p1p2 = np.array(p1) - np.array(p2)
    vector_p3p2 = np.array(p3) - np.array(p2)
    cosine_angle = np.dot(vector_p1p2, vector_p3p2) / (np.linalg.norm(vector_p1p2) * np.linalg.norm(vector_p3p2))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def visualize_result(s, e, a, b, midpoint):
    # Calculate angles for annotations
    angle_sm_mb = calculate_angle(s, midpoint, b)
    angle_em_ma = calculate_angle(e, midpoint, a)

    # Calculate lengths of MA and MB
    length_ma = np.linalg.norm(a - midpoint)
    length_mb = np.linalg.norm(b - midpoint)

    plt.figure(figsize=(8, 8))
    plt.plot([s[0], e[0]], [s[1], e[1]], 'k-', label='SE')
    plt.plot([midpoint[0], a[0]], [midpoint[1], a[1]], 'b-', label=f'MA (Length: {length_ma:.2f})')
    plt.plot([midpoint[0], b[0]], [midpoint[1], b[1]], 'g-', label=f'MB (Length: {length_mb:.2f})')

    # Mark points with coordinates
    for point, label in zip([s, e, midpoint, a, b], ['S', 'E', 'M', 'A', 'B']):
        plt.text(point[0], point[1], f'{label} {np.round(point, 2)}', fontsize=12, verticalalignment='bottom' if label in ['S', 'E'] else 'top')

    # Adjust angle annotations to avoid overlap
    plt.annotate(f'SMB={angle_sm_mb:.2f}°', (midpoint[0], midpoint[1]), textcoords="offset points", xytext=(50,40), ha='center')
    plt.annotate(f'EMA={angle_em_ma:.2f}°', (midpoint[0], midpoint[1]), textcoords="offset points", xytext=(-50,40), ha='center')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()

def count_matches(source_strings, target_string):
    """
    Counts how many elements in source_strings contain target_string.

    :param source_strings: List of strings to search through.
    :param target_string: String to search for within source_strings.
    :return: Integer representing the number of matches.
    """
    count = sum(target_string in s for s in source_strings)
    return count