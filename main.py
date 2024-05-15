%%writefile test2_app.py
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
    st.set_page_config(page_title='Diffusion Model Generator', layout='centered')
    st.image('/content/fig/planter_logo.png', use_column_width=True)
    st.caption('*“Blooming every doorstep to your place”*')  # Italicized caption

    # Main vertical boxes (containers)
    top_container = st.container()
    with top_container:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader('Upload Image')
            uploaded_file = st.file_uploader("Choose an image (optional)", type=['jpg', 'png', 'jpeg'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)  # Display uploaded image immediately

        with col2:
            st.subheader('Specify Your Preferences')
            light_intensity = st.selectbox("Light Intensity", ["Low", "Medium", "High"])
            plant_size = st.selectbox("Plant Size", ["Small", "Medium", "High"])
            plant_category = st.selectbox("Plant Category", ["Herb", "Tree", "Flower"])
            care_difficulty = st.selectbox("Level of Difficulty to Take Care", ["Easy", "Medium", "Hard"])

    # Prompt input below image and selection boxes
    prompt_container = st.container()
    with prompt_container:
        st.subheader('Enter Your Prompt')
        prompt = st.text_area("", 'Input the prompt desired', help="Enter a descriptive prompt for the image generation.")

    # Bottom container for displaying the generated image
    bottom_container = st.container()
    with bottom_container:
        if st.button("Generate Image"):
            # Combine user inputs into a full prompt
            full_prompt = f"{prompt}. Light Intensity: {light_intensity}, Plant Size: {plant_size}, Plant Category: {plant_category}, Care Difficulty: {care_difficulty}."
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
