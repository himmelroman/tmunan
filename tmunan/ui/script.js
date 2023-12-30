function transitionImage(imageId){

    // Get image frames
    const topImage = document.getElementById('topImage');
    const bottomImage = document.getElementById('bottomImage');

    // Prepare new image url
    const image_url = `http://localhost:8000/images/${imageId}`

    // Set the new image in bottom frame
    bottomImage.src = image_url;

    // Fade out top image
    topImage.style.opacity = '0%';
    topImage.style.transition = 'opacity 3s ease-in-out';

     // After the transition completes
    setTimeout(() => {
        topImage.src = image_url;           // Switch places - bottom (new) image is now at top frame
        topImage.style.opacity = '100%';    // Fade it back in
    }, 3000);
}
