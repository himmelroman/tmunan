let api_address = 'localhost'
let animationSpeed = 1;
let transitionInterval = 2;
let inTransition = false;
const imageQueue = [];

window.onload = function () {

    // init api address
    api_address = window.location.hostname;

    // Init config panel
    const config_address = document.getElementById('config_api_address');
    const config_animation_speed = document.getElementById('config_animation_speed');
    const config_transition_interval = document.getElementById('config_transition_interval');
    config_address.value = api_address;
    config_animation_speed.value = animationSpeed;
    config_transition_interval.value = transitionInterval;
};

function updateConfig() {

    // Get config inputs
    const config_address = document.getElementById('config_api_address');
    const config_animation_speed = document.getElementById('config_animation_speed');
    const config_transition_interval = document.getElementById('config_transition_interval');

    // Update variables
    api_address = config_address.value;
    animationSpeed = config_animation_speed.value;
    transitionInterval = config_transition_interval.value;
}

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
    const new_image_url = `http://${api_address}:8080/api/images/${dequeueImage()}`;
    const newImage = document.createElement('img')

    // Get old image
    const oldImage   = imageContainer.querySelector('img')

    // Append new image
    newImage.src = new_image_url;
    imageContainer.insertBefore(newImage, imageContainer.firstChild);

    // Wait until image is loaded
    newImage.onload = function(e){

        // Start fading out old image
        oldImage.style.opacity = '0%';
        oldImage.style.transition = `opacity ${animationSpeed}s linear` // ease-in-out`;

        // After the transition completes
        oldImage.addEventListener("transitionend", () => {
            oldImage.remove();
        })
    }
}

advanceSlideshow();
