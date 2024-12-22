import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

palette = [(0.00784313725490196, 0.24313725490196078, 1.0),
           (0.10196078431372549, 0.788235294117647, 0.2196078431372549),
           (1.0, 0.48627450980392156, 0.0)]

x = np.array([2, 3])
y = np.array([10, 1])

# Adjusting the plot scale and settings for better clarity
plt.figure(figsize=(10, 6))
plt.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy',
           scale=1, color=palette[0], width=0.01, label='Vector x')
plt.quiver(0, 0, y[0], y[1], angles='xy', scale_units='xy',
           scale=1, color=palette[2], width=0.01, label='Vector y')

# Calculating fewer interpolation points for clearer visualization
fewer_a_values = np.linspace(0, 1, 10)  # Use fewer points for clarity
interpolated_points = [(1-a)*x + a*y for a in fewer_a_values[1:-1]]

for a in fewer_a_values[1:int(len(fewer_a_values)/2)]:
    interpolated_point = (1-a)*x + a*y
    plt.quiver(0, 0, interpolated_point[0], interpolated_point[1], angles='xy',
               scale_units='xy', scale=1, color=palette[1], alpha=1.0, width=0.01,
               hatch='|||', facecolor='none')

# Tagging one specific green arrow with \overline{M_r}
# Choose the middle point for clarity
mid_point = interpolated_points[len(interpolated_points)//2]
plt.quiver(0, 0, mid_point[0], mid_point[1], angles='xy',
           scale_units='xy', scale=1, color=palette[1], alpha=1.0, width=0.01)
plt.text(mid_point[0]*0.98, mid_point[1]*1.1,
         r'$\overline{M_{r-1}}$', fontsize=24)

# Adding labels beside red and blue arrows
plt.text(x[0]*0.98, x[1]*1.05,
         r'$\overline{M_{r-2}}$', fontsize=24)
plt.text(y[0]*0.88, y[1]*1.25, r'$M_{r-1}$', fontsize=24)

# Drawing angle arc for \theta
arc_radius = 5
start_angle = np.degrees(np.arctan2(y[1], y[0]))
end_angle = np.degrees(np.arctan2(mid_point[1], mid_point[0]))
arc = patches.Arc((0, 0), arc_radius, arc_radius, angle=0, theta1=min(
    start_angle, end_angle), theta2=max(start_angle, end_angle), color='black', linewidth=2)
plt.gca().add_patch(arc)

# Add angle label
theta_pos = (arc_radius / 1.5) * np.array([np.cos(np.radians(
    (start_angle + end_angle) / 2)), np.sin(np.radians((start_angle + end_angle) / 2))])
plt.text(theta_pos[0]*0.9, theta_pos[1]*0.85, r'$\theta$',
         fontsize=24, ha='center', va='center')

# plt.xlabel('X coordinate')
# plt.ylabel('Y coordinate')
# plt.title('Vectors x and y from Origin with Fewer Interpolations')
# plt.legend()
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
# plt.grid(True)
# plt.axis('equal')
plt.xlim(-0.5, 10.5)
plt.ylim(-0.5, 4)

# plt.show()
plt.savefig(
    f"./save/Drift.png",
    bbox_inches='tight',
    dpi=300)
