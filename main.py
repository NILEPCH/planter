

import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline
from torch import autocast
import torch

class StableDiffusionLoader:
    def __init__(self, prompt, pretrain_pipe='CompVis/stable-diffusion-v1-4'):
        self.prompt = prompt
        self.pretrain_pipe = pretrain_pipe
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == 'cpu':
            raise MemoryError('GPU needed for inference')

    def generate_image_from_prompt(self, save_location='prompt.jpg', use_token=False, verbose=False):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.pretrain_pipe, revision="fp16", torch_dtype=torch.float16, use_auth_token=use_token
        ).to(self.device)

        with autocast(self.device):
            image = pipe(self.prompt)[0]['sample'][0]

        image.save(save_location)
        if verbose:
            print(f'[INFO] Saving image to {save_location}')
        return image

if __name__ == '__main__':
    st.set_page_config(page_title='Diffusion Model Generator')
    st.image('/content/fig/planter_logo.png')
    st.caption('“ Blooming every doorstep to your place “')

    # Main vertical boxes (containers)
    top_container = st.container()
    with top_container:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader('Upload Image and Enter Prompt')
            uploaded_file = st.file_uploader("Choose an image (optional)", type=['jpg', 'png', 'jpeg'])
            prompt = st.text_area('Input the prompt desired')

        with col2:
            st.subheader('Specify Your Preferences')
            light_level = st.selectbox("Light Level", ["Low", "Medium", "High"])
            maintenance_level = st.selectbox("Maintenance Level", ["Low", "Medium", "High"])
            plant_size = st.selectbox("Plant Size", ["Small", "Medium", "Large"])
            plant_category = st.selectbox("Plant Category", ["Indoor", "Outdoor", "Flowering", "Non-flowering"])
            theme = st.selectbox("Theme", ["Nature", "Abstract", "Urban", "Fantasy"])

    # Bottom container for displaying the generated image
    bottom_container = st.container()
    with bottom_container:
        if st.button("Generate Image"):
            # Concatenate preferences with the prompt
            full_prompt = f"{prompt}. Light Level: {light_level}, Maintenance Level: {maintenance_level}, Plant Size: {plant_size}, Plant Category: {plant_category}, Theme: {theme}"
            if full_prompt:
                with st.spinner('Generating image based on prompt and preferences...'):
                    sd = StableDiffusionLoader(full_prompt)
                    try:
                        image_path = sd.generate_image_from_prompt()
                        image = Image.open(image_path)
                        st.image(image, caption='Generated Image', use_column_width=True)
                        st.success('Image generated successfully!')
                    except Exception as e:
                        st.error(f"Failed to generate image: {str(e)}")
            else:
                st.warning("Please enter a prompt and select preferences to generate an image.")

