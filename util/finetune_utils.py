import torch
from tqdm import tqdm
from kornia.morphology import dilation
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image

def finetune_decoder(config, model, render_output, inpaint_output, custom_mode, n_steps=100):
    params = [{"params": model.vae.decoder.parameters(), "lr": config["decoder_learning_rate"]}]
    optimizer = torch.optim.Adam(params)
    if custom_mode == "wonderjourney":
        decoder_ft_mask = render_output["inpaint_mask"].detach()
        rendered_image = render_output["rendered_image"].detach()
    elif custom_mode == "wonderjourney_576_1024":
        decoder_ft_mask = render_output["inpaint_mask"].detach()
        decoder_ft_mask = ToTensor()( ToPILImage()(decoder_ft_mask[0]).resize((512,288), Image.NEAREST) ).unsqueeze(0).to("cuda")[:, 0:1, :, :] # resize 到288 * 512
        rendered_image = render_output["rendered_image"].detach()
        rendered_image = ToTensor()( ToPILImage()(rendered_image[0]).resize((512,288), Image.NEAREST) ).unsqueeze(0).to("cuda") # resize 到288 * 512
    else:
        raise("no implementation for given custom_mode")
    
    ToPILImage()(decoder_ft_mask[0]).save(model.run_dir / 'images' / 'decoder_ft_mask.png')
    if config['dilate_mask_decoder_ft'] > 1:
        decoder_ft_mask_dilated = dilation(decoder_ft_mask, torch.ones(config['dilate_mask_decoder_ft'], config['dilate_mask_decoder_ft']).to('cuda'))
    else:
        decoder_ft_mask_dilated = decoder_ft_mask
    ToPILImage()(decoder_ft_mask_dilated[0]).save(model.run_dir / 'images' / 'decoder_ft_mask_dilated.png')
    for _ in tqdm(range(n_steps), leave=False):
        optimizer.zero_grad()
        loss = model.finetune_decoder_step(
            inpaint_output["inpainted_image"].detach(),
            inpaint_output["latent"].detach(),
            rendered_image,
            decoder_ft_mask,
            decoder_ft_mask_dilated,
        )
        loss.backward()
        optimizer.step()

    del optimizer


def finetune_depth_model(config, model, target_depth, epoch, mask_align=None, mask_cutoff=None, cutoff_depth=None):
    params = [{"params": model.depth_model.parameters(), "lr": config["depth_model_learning_rate"]}]
    optimizer = torch.optim.Adam(params)

    if mask_align is None:
        mask_align = target_depth > 0

    for _ in tqdm(range(config["num_finetune_depth_model_steps"]), leave=False):
        optimizer.zero_grad()

        loss = model.finetune_depth_model_step(
            target_depth,
            model.images[epoch],
            mask_align=mask_align,
            mask_cutoff=mask_cutoff,
            cutoff_depth=cutoff_depth,
        )
        try:
            loss.backward()
            optimizer.step()
        except RuntimeError:
            print('No valid pixels to compute depth fine-tuning loss. Skip this step.')
            return
