import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

contents = ['sjc']
styles = ['constable', 'dali', 'munch']

for content in contents:
    for style in styles:
        directory = 'outputs/%s/%s/' % (content, style)
        fig, ax = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(20, 14))
        ax = ax.ravel()

        for i, jpg in enumerate(os.listdir(directory)):
            img = mpimg.imread(directory + jpg)
            ax[i].imshow(img)
            ax[i].axis('off')
            ax[i].set_title(jpg[:-4])

        plt.suptitle('%s_%s' % (content, style) + '\n{iterations}_{content-weight-blend}_{style-layer-weight-exp}_{learning-rate}', fontsize=18)
        plt.savefig('images/%s_%s.jpg' % (content, style), format='jpg', dpi=300)