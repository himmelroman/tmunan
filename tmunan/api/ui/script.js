const slideshowIntervalInput = document.getElementById('slideshow-interval');
const playButton = document.getElementById('play');
const stopButton = document.getElementById('stop-button');
const imageContainer = document.getElementById('image-container');
const promptSeqContainer = document.getElementById('prompt-seq');
const entryPromptInput = document.getElementById('add-prompt-text');
const addPromptButton = document.getElementById('add-prompt-button');
const addPromptStrength = document.getElementById('add-prompt-strength');
const addPromptNewSeq = document.getElementById('add-prompt-new-seq');

let currentIndex = 0;
let intervalId;
let imageTimerId;
let sequenceKeyImageId;


function updateImage() {

  // Clear any previous timer
  clearInterval(imageTimerId);

  // Get the display time from the input element (in seconds)
  const displayTimeInput = slideshowIntervalInput.value;

  // Fetch the next prompt text
  const currentPromptElement = promptSeqContainer.querySelectorAll('textarea')[currentIndex]
  const currentPrompt = currentPromptElement.value;

  // Check for empty prompts and handle edge cases
  const totalPrompts = promptSeqContainer.querySelectorAll('textarea').length;
  console.log(`Total prompts: ${totalPrompts}`);
  if (!currentPrompt || currentIndex >= totalPrompts) {
    currentIndex = 0; // Reset to first prompt if empty or exceeding

    if (playButton.classList.contains('active')) {
      console.warn('No prompts available, slideshow stopped.');
      clearInterval(intervalId);
      return; // Stop slideshow if no prompts
    }
  }

  // Prepare image generation instructions
  const new_seq= currentPromptElement.dataset.hasOwnProperty('new_sequence')
  const task = (new_seq? 'txt2img' : 'img2img')
  const instructions = {
      prompt: currentPrompt,
      image_id: sequenceKeyImageId,
      strength: parseFloat(currentPromptElement.dataset.strength)
  }

  // Fetch the image
  fetch(`http://localhost:8000/${task}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(instructions)
  })
    .then(response => response.json())
    .then(data => {
      if (!data.image_url) {
        console.log(data.error);
        return; // Stop execution if generation fails
      }

      // Update image_id
      if (task === 'txt2img') {
        sequenceKeyImageId = data.image_id
      }

      // Step 2: Fetch generated image as a blob
      fetch(data.image_url)
        .then(response => response.blob())
        .then(imageBlob => {

            // Create blob url
            const blobUrl = URL.createObjectURL(imageBlob);

            // Update the image container background
            imageContainer.style.backgroundImage = `url(${blobUrl})`;

            // visualize current prompt
            promptSeqContainer.querySelectorAll('.prompt-item').forEach(i => i.classList.remove('current'))
            promptSeqContainer.querySelectorAll('.prompt-item')[currentIndex].classList.add('current')

            // Increment currentIndex and update image after timer
            if (currentIndex < totalPrompts - 1) {
              currentIndex++;
            } else {
              currentIndex = 0; // Reset to first prompt if all used
            }

            // Convert display time input to milliseconds for setTimeout
            const displayTime = displayTimeInput * 1000;

            // Start the timer directly within the callback
            imageTimerId = setTimeout(() => {

              // Release the used URL after display
              URL.revokeObjectURL(blobUrl);

              updateImage();
            }, displayTime);
        })
        .catch(error => console.error(`Failed to fetch generated image: ${error}`));
    })
    .catch(error => console.error(`Image generation failed: ${error}`));
}

imageContainer.addEventListener('load', () => {
  // Clear any previous timer
  clearInterval(imageTimerId);

  // Get the display time from the input element (in seconds)
  const displayTimeInput = slideshowIntervalInput.value;

  // Convert display time input to milliseconds for setTimeout
  const displayTime = displayTimeInput * 1000;

  // Start timer directly in the load event handler
  imageTimerId = setTimeout(() => {
    // Update image after timer elapses
    updateImage();
  }, displayTime);
});

playButton.addEventListener('click', function () {
  if (intervalId) {
    clearInterval(intervalId);
  }

  intervalId = setInterval(updateImage, slideshowIntervalInput.value * 1000);
  updateImage(); // Start with the first prompt
});

stopButton.addEventListener('click', () => {
  // Clear the existing timer to stop the slideshow
  clearInterval(intervalId);
});

addPromptButton.addEventListener('click', () => {
  const newPromptContainer = document.createElement('div');
  newPromptContainer.className = "prompt-item"
  const newPromptTextarea = document.createElement('textarea');
  const strengthLabel = document.createElement('label');
  const removePromptButton = document.createElement('button');
  removePromptButton.innerText = "X"

  // Set attributes and styles for the new elements
  newPromptTextarea.innerHTML = entryPromptInput.value;
  strengthLabel.innerHTML = addPromptStrength.value;
  if (addPromptNewSeq.checked) {
      newPromptTextarea.className = 'new_sequence';
      newPromptTextarea.dataset.new_sequence = 'true';
  } else {
      newPromptTextarea.dataset.strength = addPromptStrength.value;
  }

  // Clear entry controls
  entryPromptInput.value = '';
  addPromptNewSeq.checked = false;

  // Add functionality to remove button
  removePromptButton.addEventListener('click', () => {
    // Remove the parent element containing the input and button
    newPromptContainer.parentNode.removeChild(newPromptContainer);
  });

  // Append new elements to the container
  newPromptContainer.appendChild(newPromptTextarea);
  newPromptContainer.appendChild(strengthLabel);
  newPromptContainer.appendChild(removePromptButton);
  document.querySelector('#prompt-seq').appendChild(newPromptContainer);
});
