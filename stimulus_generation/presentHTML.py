"""
HTML summary pages for the synthesized stimuli.

* make_presnetHTML  -> one ``present.html`` per stimulus folder: every level's
                      GIFs with their loss, the loss plot and the 3D PCA plots.
* generate_htmlFile -> one overview ``present.html`` for all stimulus folders.
"""

import os


def make_presnetHTML(num_levels, merged_path, loss_atSave, nex=8):
    """Write ``present.html`` inside `merged_path`.

    Parameters
    ----------
    num_levels : int
        Number of saved levels (rows of GIFs).
    merged_path : str
        Stimulus folder that contains the GIFs and plots.
    loss_atSave : list
        Loss value at each saved level (shown under each GIF).
    nex : int
        Number of stimuli per level (2 = one pair).
    """
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{os.path.basename(merged_path)}</title>
        <style>
            #imageContainer {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
            }}

            .levelContainer {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-around;
                margin-bottom: 20px;
            }}

            .levelContainer img {{
                width: 100px;
                height: 100px;
                margin: 5px;
            }}

            .levelContainer .imageWrapper {{
                text-align: center;
                margin: 10px;
            }}

            .pngContainer {{
                text-align: center;
            }}

            .pngContainer img {{
                width: auto;
                height: auto;
                max-width: 100%;
                max-height: 100%;
            }}
        </style>
    </head>
    <body>

    <h1 style="text-align: center; font-weight: bold;">{os.path.basename(merged_path)}</h1>

    <div id="imageContainer">
    """

    # One row per level, one GIF per stimulus
    for level in range(num_levels):
        html_content += f'\n    <div class="levelContainer">\n'
        for imageIndex in range(nex):
            html_content += f'        <div class="imageWrapper">\n'
            html_content += f'            <img src="level_{level}_basic_animation_{imageIndex}.gif" alt="Image">\n'
            rounded_loss = round(loss_atSave[level].item(), 2)
            html_content += f'            <p>L{level}_img{imageIndex}_Loss{rounded_loss}</p>\n'
            html_content += '        </div>\n'
        html_content += '    </div>\n'

    # Loss plot of the target-pair optimization
    html_content += f"""
    </div>
    <div id="imageContainer">
    <div class="levelContainer pngContainer">
        <div class="imageWrapper">
            <img src="losses.png" alt="Losses Image">
            <p>Step 1: Training the stimuli to have distinct response in the activation layer</p>
        </div>
    </div>
    """

    # Loss plot of the graded series (only present if it was saved)
    html_content += f"""
    </div>
    <div id="imageContainer">
    <div class="levelContainer pngContainer">
        <div class="imageWrapper">
            <img src="losses_G1_L1.png" alt="Losses Image">
            <p>Step 2: losses through moving from full noise toward the fully trained stimuli</p>
        </div>
    </div>
    """

    # Static 3D PCA plot
    html_content += f"""
    </div>
    <div id="imageContainer">
    <div class="levelContainer pngContainer">
        <div class="imageWrapper">
            <img src="3D_Visualization_videos.png" alt="3D_Visualization Image">
            <p>3D_Visualization.png</p>
        </div>
    </div>
    """

    # Interactive 3D PCA plot
    html_content += f"""
    </div>
    <div id="imageContainer">
    <div class="levelContainer pngContainer">
        <div class="imageWrapper">
            <iframe src="3D_interactive_plot.html" style="width: 800px; height: 600px; border: none;"></iframe>
            <p>Interactive 3D Plot</p>
        </div>
    </div>
    """

    html_content += """
    </div>

    </body>
    </html>
    """

    present_html_path = os.path.join(merged_path, 'present.html')
    with open(present_html_path, 'w') as html_file:
        html_file.write(html_content)

    print(f"HTML file saved at: {present_html_path}")


def generate_htmlFile(output_directory):
    """Write an overview ``present.html`` in `output_directory` showing, for
    each stimulus sub-folder, its first two GIFs and its loss plot."""
    html_file = os.path.join(output_directory, "present.html")

    subfolder_directory = output_directory

    os.makedirs(output_directory, exist_ok=True)

    with open(html_file, "w") as f:
        f.write("<html>\n<head>\n<title>GIF and Losses Presentation</title>\n</head>\n<body>\n")
        f.write("<h1>GIF and Losses Presentation</h1>\n")
        f.write("<style>img.square { width: 150px; height: 150px; object-fit: cover; margin-right: 10px; }</style>\n")

        for subfolder in os.listdir(subfolder_directory):
            subfolder_path = os.path.join(subfolder_directory, subfolder)
            if os.path.isdir(subfolder_path):
                gif_files = [file for file in os.listdir(subfolder_path) if file.endswith(".gif")]
                losses_file = os.path.join(subfolder_path, "losses.png")

                print(f"Checking subfolder: {subfolder}")
                print(f"GIF files found: {gif_files}")
                print(f"Losses file found: {'Yes' if os.path.exists(losses_file) else 'No'}")

                if len(gif_files) >= 2:
                    f.write(f"<h2>{subfolder}</h2>\n")
                    f.write("<div style='display: flex;'>\n")

                    for gif in gif_files[:2]:
                        gif_path = os.path.join(subfolder, gif)  # relative path for HTML
                        f.write(f"<img src='{gif_path}' class='square'>\n")

                    if os.path.exists(losses_file):
                        losses_relative_path = os.path.join(subfolder, "losses.png")
                        f.write(f"<img src='{losses_relative_path}' style='margin-right: 10px;'>\n")

                    f.write("</div>\n")
                    f.write("<br>\n")

        f.write("</body>\n</html>")

    print(f"HTML file saved at '{html_file}'.")
