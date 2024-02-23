# from .srn import SRNDataset
# from .co3d import CO3DDataset
from .mip360 import MIP360Dataset

def get_dataset(cfg, name):
    # if cfg.data.category == "cars" or cfg.data.category == "chairs":
    #     return SRNDataset(cfg, name)
    # elif cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
    #     return CO3DDataset(cfg, name)
    if cfg.data.category == "mip360":
        return MIP360Dataset(cfg,name)

# @torch.no_grad()
# def readImages(renders_dir, gt_dir):
#     renders = []
#     gts = []
#     image_names = []
#     for fname in os.listdir(renders_dir):
#         render = Image.open(renders_dir / fname)
#         gt = Image.open(gt_dir / fname)
#         renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
#         gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
#         image_names.append(fname)
#     return renders, gts, image_names