import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import ipdb
import time


class VIS_TOOLS():
    def __init__(self,root='/home/wuyang/Pictures/GMDA/score_maps/'):
        self.root = root
        if not os.path.exists(root):
            os.makedirs(root)


    def show_adj(self, Adj, name='adj',id=0, num_nodes=99999):
        root = os.path.join(self.root, name)
        if not os.path.exists(root):
            os.makedirs(root)


        if Adj.size(0)<num_nodes:
            Adj = Adj.detach().cpu().numpy()
            plt.imshow(Adj)
            plt.savefig(root+'/{}_nodes.png'.format(id))



    def save_feat(self, feats, id=0, folder_name='cls',to_cpu=True):
        feat_root = self.root + 'features/' + folder_name +'/'

        if not os.path.exists(feat_root):
            os.makedirs(feat_root)

        if type(feats) == list and to_cpu:
            for i, feat in enumerate(feats):
                feats[i] = feat.detach().cpu()
        elif to_cpu:
            feats = feats.detach().cpu()
        path = feat_root  +  '/{}.pt'.format(id)

        torch.save(feats, path)

    def load_feat(self, cnt=0, name='cls'):
        feat_root = self.root + '/features/'
        if not os.path.exists(feat_root):
            os.makedirs(feat_root)

        path = feat_root + name + '/' + '_{}.pt'.format(cnt)
        return torch.load(path)


    def debug_T_SNE(self, prototype, name='tsne_prototype', exit=False):
        root = self.root + '/{}/'.format(name)
        if not os.path.exists(root):
            os.makedirs(root)
        num_classes = prototype.size(0)
        TSNE_embedded = TSNE(n_components=2).fit_transform(prototype.cpu().numpy())
        legend = []
        legend_name = []

        def get_cmap(n, name='hsv'):
            '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
            RGB color; the keyword argument name must be a standard mpl colormap name.'''
            return plt.cm.get_cmap(name, n)

        cmap = get_cmap(num_classes)

        for i in range(num_classes):
            legend.append(plt.scatter(TSNE_embedded[i, 0], TSNE_embedded[i, 1], color=cmap(i), s=20))
            legend_name.append('class_{}'.format(i + 1))
        plt.legend(handles=legend, labels=legend_name, loc=1, prop={'size': 8})

        # plt.set_cmap('rainbow')
        plt.savefig(root + 'tsne.png', dpi=600)
        plt.close()
        if exit:
            os._exit(0)


    def draw_activation_maps(self, act_maps, feat_level, name='activation_maps', exit=False, throd = False):
        target_size = (1, 1, 800, 1333)
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        root = self.root + '/{}/'.format(name)
        if not os.path.exists(root):
            os.makedirs(root)

        # for img in range(act_maps.size(0)):
        #     for cls, show_map in enumerate(act_maps[img]):
        #         if throd:
        #             show_map = show_map * (show_map>throd).float()
        #
        #         show_map = show_map.view(1, 1, show_map.size(0), show_map.size(1)).cpu()
        #         show_map = F.interpolate(show_map, size=(800, 1333), mode='bilinear').squeeze()
        #         show_map = show_map.numpy()
        #         show_map[0,0]=1
        #         fig = plt.figure(tight_layout=True)
        #         ax = fig.add_subplot(111)
        #         im = ax.imshow(show_map, 'jet')
        #         ax.axis('off')
        #         divider = make_axes_locatable(ax)
        #         cax = divider.append_axes('right', size='2%', pad=0.04)
        #         cbar = plt.colorbar(im,
        #                             cax=cax,
        #                             extend='both',
        #                             extendrect=True,
        #                             ticks=list(np.linspace(0,1,11)),
        #                             )
        #         cbar.outline.set_visible(False)
        #         cbar.ax.tick_params(labelsize=8,
        #                             width=0,
        #                             length=0,
        #                             pad=1, )
        #         # save image
        #         fig.savefig(root + 'image-{}_level-{}_class-{}_map.png'.format(img, feat_level, cls),
        #                     bbox_inches='tight',
        #                     pad_inches=0.2,
        #                     transparent=True,
        #                     dpi=300)
        #
        #         plt.close()
        img = 0
        for cls, show_map in enumerate(act_maps[img]):
            if throd:
                show_map = show_map * (show_map>throd).float()

            show_map = show_map.view(1, 1, show_map.size(0), show_map.size(1)).cpu()
            show_map = F.interpolate(show_map, size=(800, 1333), mode='bilinear').squeeze()
            show_map = show_map.numpy()
            show_map[0,0]=1
            fig = plt.figure(tight_layout=True)
            ax = fig.add_subplot(111)
            im = ax.imshow(show_map, 'jet')
            ax.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='2%', pad=0.04)
            cbar = plt.colorbar(im,
                                cax=cax,
                                extend='both',
                                extendrect=True,
                                ticks=list(np.linspace(0,1,11)),
                                )
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(labelsize=8,
                                width=0,
                                length=0,
                                pad=1, )
            # save image
            fig.savefig(root + 'image-{}_level-{}_class-{}_map.png'.format(img, feat_level, cls),
                        bbox_inches='tight',
                        pad_inches=0.2,
                        transparent=True,
                        dpi=300)

            plt.close()

        if exit:
            os._exit(0)

'''
debug prototype similarity
'''

# self.target_prototype = torch.zeros(self.num_classes_bg, self.prototype_channel).cuda() # For debuging

# ipdb.set_trace()
# for i, feat in enumerate(self.target_prototype):
#
#     if (feat!=0).sum()==0:
#         self.target_prototype[i]=prototype_buffer_batch[i]
#
# print((self.target_prototype.sum(dim=-1)!= 0))
# print(self.target_prototype)
# if   (self.target_prototype.sum(dim=-1)!= 0).sum()>7:
# # if False not in (self.target_prototype.sum(dim=-1)!= 0):
#     self.debugger.save_feat(self.prototype, folder_name='prototype', id='aligned_source')
#     self.debugger.save_feat(self.target_prototype,folder_name='prototype', id= 'aligned_target')
#     ipdb.set_trace()