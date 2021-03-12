import numpy as np

# color pallete from https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=10
link_rgba = np.array([
    # == first 8
    [166, 206, 227, 255], # Light Blue
    [31, 120, 180, 255],  # Dark Blue
    [178, 223, 138, 255], # Light Green
    [51, 160, 44, 255],   # Dark Green
    [251, 154, 153, 255], # Pink
    [227, 26, 28, 255],   # Red
    [253, 191, 111, 255], # Yellow
    [255, 127, 0, 255],   # Orange
    # == last 2
    [202, 178, 214, 255], # Light Purple
    [106, 61, 154, 255]   # Dark Purple
], dtype=np.float32)

link_rgba = link_rgba / 255


def get_link_color(idx):
    return link_rgba[int(idx)]



# plot color pallete, save as above
plot_color = ["#a6cee3",
"#1f78b4",
"#b2df8a",
"#33a02c",
"#fb9a99",
"#e31a1c",
"#fdbf6f",
"#ff7f00",
"#cab2d6",
"#6a3d9a"]

def get_plot_color(idx):
    return plot_color[idx]
    
if __name__ == "__main__":
    print(link_rgba)
