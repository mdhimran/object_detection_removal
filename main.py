from imports import *
from utils import *

def main():
    st.title("Object Detection and Removal")

    st.write("Upload an image to perform object detection and inpainting.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_tensor = transform_image(image)
        model = get_model()

        # Create three columns for images
        col1, col2= st.columns(2)

        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Detect Objects'):
            predictions = detect_objects(model, image_tensor)
            st.session_state.predictions = predictions
            st.session_state.detection_done = True

            with col2:
                # st.write("Detected objects:")
                detected_image = visualize_detection(image.copy(), predictions)
                st.image(detected_image, caption='Detected Image', use_column_width=True)

        if 'detection_done' in st.session_state and st.session_state.detection_done:
            if st.button('Remove Objects'):
                if 'predictions' in st.session_state:
                    predictions = st.session_state.predictions
                    # st.write("Inpainting objects...")
                    Removal_image = inpaint_objects(image, predictions)

                    with col2:
                        st.image(Removal_image, caption='Removal Image', use_column_width=True)
                else:
                    st.write("Please detect objects first.")

if __name__ == "__main__":
    main()