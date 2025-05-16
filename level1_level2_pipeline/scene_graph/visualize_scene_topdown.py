import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Load JSON data from file
file_path = "./toy_scene_3_dir.json"
# file_path = './toy_scene_2.json'
# file_path = './toy_scene_floating.json'
with open(file_path, "r") as f:
    data = json.load(f)

# Parse the object instances from the JSON file
objects = []
for obj in data["object_instances"]:
    x_centroid = obj["centroid_translation"]["x"]
    y_centroid = obj["centroid_translation"]["y"]
    bbox_width = obj["bbox"]["x_length"]
    bbox_height = obj["bbox"]["y_length"]
    objects.append(
        {
            "name": obj["name"],
            "centroid": (x_centroid, y_centroid),
            "bbox": (bbox_width, bbox_height),
        }
    )

# Create a plot with center point annotations
fig, ax = plt.subplots()

# Plot each object with its center point only
for obj in objects:
    x_centroid, y_centroid = obj["centroid"]
    bbox_width, bbox_height = obj["bbox"]

    # Calculate the bottom-left corner of the bbox (not used now)
    bottom_left_x = x_centroid - bbox_width / 2
    bottom_left_y = y_centroid - bbox_height / 2

    # Add a rectangle for the bbox without corner points
    rect = patches.Rectangle(
        (bottom_left_x, bottom_left_y),
        bbox_width,
        bbox_height,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)

    # Mark the center point
    ax.plot(x_centroid, y_centroid, "bo")  # center point in blue

    # Annotate the center point only
    ax.text(
        x_centroid,
        y_centroid,
        f"({x_centroid:.1f}, {y_centroid:.1f})",
        fontsize=8,
        color="blue",
    )

# Set labels and title
ax.set_xlabel("X axis (front)")
ax.set_ylabel("Y axis (left)")
ax.set_title("Top View of Objects with Center Points")

# Adjust the aspect ratio
ax.set_aspect("equal")

# Show the plot
# plt.show()
output_image_path = "./replica_apt_0_topdown.png"
fig.savefig(output_image_path)
