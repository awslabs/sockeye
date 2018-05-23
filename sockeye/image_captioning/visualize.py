# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Visualize the checkpoints of the model: image, ground truth caption and
predicted caption.
"""
import argparse
import os

try:  # Try to import pillow
    from PIL import Image  # pylint: disable=import-error
except ImportError as e:
    raise RuntimeError("Please install pillow.")

try:  # Try to import matplotlib
    import matplotlib  # pylint: disable=import-error
except ImportError as e:
    raise RuntimeError("Please install matplotlib.")
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def format_text_for_visualization(c, n):
    c = c.split(" ")
    c[0] = c[0].title()
    out = ""
    for j in range(0, len(c)):
        out += c[j]
        if j == len(c)-1:
            out += "."
        else:
            if (j + 1) % n == 0:
                out += "\n"
            else:
                out += " "
    return out

def main():
    params = argparse.ArgumentParser(
        description="CLI to visualize the captions along with images and "
                    "ground truth."
    )
    params.add_argument("-d", "--image-root",
                        help="Absolute path of the dataset root where the "
                             "images are stored.")
    params.add_argument("-i", "--source",
                        help="File containing the images or features used to "
                             "generate the captions.")
    params.add_argument("-c", "--prediction",
                        help="File containing the captions. Each line "
                             "corresponds to a line in the source.")
    params.add_argument("-a", "--ground-truth",
                        default=None,
                        help="File file containing the ground-truth captions "
                             "(optional).")
    params.add_argument("-s", "--save-to-folder",
                        default=None,
                        help="Folder to save the visualizations.")
    params.add_argument("-si", "--skip-images",
                        default=2,
                        help="Number of images to skip for visualization.")
    params.add_argument("-nc", "--number-of-columns",
                        default=4,
                        help="Number of columns in the subplot (better if even "
                             "number).")
    args = params.parse_args()

    skip = args.skip_images
    N = M = args.number_of_columns

    # adjust this if visualization is not nice
    len_newline = 9
    fontsize = 10
    figsize = (30, 20)

    # Collect results in a better data structure (dict)
    # * Read predictions and image dir
    fs = open(args.source)
    fc = open(args.prediction)
    predictions = {}
    for s, c in zip(fs.readlines(), fc.readlines()):
        predictions[s] = c  # just keep one sentence
    fs.close()
    fc.close()
    # * Read ground truth optionally
    ground_truth = {}
    if args.ground_truth is not None:
        fgt = open(args.ground_truth)
        fs = open(args.source)
        for s, gt in zip(fs.readlines(), fgt.readlines()):
            if s in ground_truth:
                ground_truth[s].append(gt)
            else:
                ground_truth[s] = [gt]
        fgt.close()
    fs.close()

    # Prepare output folder, if needed
    if args.save_to_folder is not None:
        fontsize = 15
        if not os.path.exists(args.save_to_folder):
            os.makedirs(args.save_to_folder)

    # Visualization
    plt.ioff()
    fig, axs = plt.subplots(N, M, figsize=figsize)
    fig.tight_layout()
    i = 0
    ii = 1
    for s in predictions.keys():  # Go over images (dict[image]=caption)
        if ii%skip==0: # maybe you do not want to display all images
            c = predictions[s]
            if len(ground_truth)>0:
                gts = ground_truth[s] # list
            s = s.split("\n")[0]
            c = c.split("\n")[0]
            # Display image
            image = Image.open(os.path.join(args.image_root, s))
            if 'RGB' not in image.mode:
                axs[i//N%M, i%N].imshow(image, cmap='gray')
            else:
                axs[i//N%M, i%N].imshow(image)
            # Display predicted caption
            axs[i//N%M, i%N].axis("off")
            axs[(i+1)//N%M, (i+1)%N].text(0, 0.9,
                                  format_text_for_visualization(c, len_newline),
                                  fontsize=fontsize,
                                  bbox={'facecolor': 'white',
                                        'alpha': 0.85,
                                        'pad': 2})
            # Display ground-truth caption(s) optionally
            if len(ground_truth)>0:
                gt_vis = ""
                for j, gt in enumerate(gts):
                    gt = gt.split("\n")[0]
                    gt_vis += \
                        "* " + format_text_for_visualization(gt, len_newline) \
                        + "\n"
                axs[(i+1)//N%M, (i+1)%N].text(0, 0, gt_vis,
                                              fontsize=fontsize,
                                              bbox={'facecolor': 'green',
                                                    'alpha': 0.3,
                                                    'pad': 2})
            axs[(i+1)//N%M, (i+1)%N].axis("off")
            i += 2

            # Show or save to disk
            if i % (N * M) == 0:
                if args.save_to_folder is None:
                    plt.show()
                else:
                    plt.savefig(os.path.join(args.save_to_folder,
                                             str(ii).zfill(6) + '.png'),
                                bbox_inches='tight')
                i = 0
                # Reset axes, clean up
                for k in range(N):
                    for j in range(M):
                        axs[k, j].cla()
                        axs[k, j].axis("off")
        ii += 1


if __name__ == "__main__":
    main()


