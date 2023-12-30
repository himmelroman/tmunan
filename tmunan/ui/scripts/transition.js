let transitionSpeed = 1;
let transitionInterval = 2;
let inTransition = false;
const imageQueue = [];

function queueImage(imageId) {
    // console.log(`Enqueuing image: ${imageId}`)
    imageQueue.push(imageId);
}

function dequeueImage() {
    return imageQueue.shift();
}

function advanceSlideshow() {

    if (inTransition) {
        console.log('Still in transition...')
        setTimeout(advanceSlideshow, 1000);
        return
    }

    if (imageQueue.length === 0) {
        console.log('Waiting for next image to become available...')
        setTimeout(advanceSlideshow, 1000);
        return
    }

    // Transition to next image
    doTransition();

    // Schedule next transition
    setTimeout(advanceSlideshow, transitionInterval * 1000);
}

function doTransition(){

    // Get container
    const imageContainer = document.getElementById('imageContainer');


    // Create new image
    const new_image_url = `http://localhost:8080/images/${dequeueImage()}`;
    const newImage = document.createElement('img')
    newImage.src = new_image_url;

    // Append new image
    const oldImage = imageContainer.querySelector('img')
    imageContainer.insertBefore(newImage, imageContainer.firstChild);

    // Fade out top image
    oldImage.style.opacity = '0%';
    oldImage.style.transition = `opacity ${transitionSpeed}s linear` // ease-in-out`;

    // After the transition completes
    oldImage.addEventListener("transitionend", () => {
        oldImage.remove();
    })
}

advanceSlideshow();
