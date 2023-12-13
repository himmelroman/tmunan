const promptsContainer = document.getElementById('prompts-container');
const slideshowIntervalInput = document.getElementById('slideshow-interval');
const playButton = document.getElementById('play');
const stopButton = document.getElementById('stop-button');
const imageContainer = document.getElementById('image-container');
const entryPromptInput = document.querySelector('#prompts-container input');
const addPromptButton = document.getElementById('add-prompt');
const removeButtons = document.querySelectorAll('#prompts-container > div:not(:first-child) button');

let currentIndex = 0;
let intervalId;
let imageTimerId;


function updateImage() {
  // Clear any previous timer
  clearInterval(imageTimerId);

  // Get the display time from the input element (in seconds)
  const displayTimeInput = slideshowIntervalInput.value;

  // Fetch the next prompt text
  const currentPrompt = promptsContainer.querySelectorAll('.prompt-container input')[currentIndex].value;

  // Check for empty prompts and handle edge cases
  const totalPrompts = promptsContainer.querySelectorAll('.prompt-container input').length;
  console.log(`Total prompts: ${totalPrompts}`);
  if (!currentPrompt || currentIndex >= totalPrompts) {
    currentIndex = 0; // Reset to first prompt if empty or exceeding

    if (playButton.classList.contains('active')) {
      console.warn('No prompts available, slideshow stopped.');
      clearInterval(intervalId);
      return; // Stop slideshow if no prompts
    }
  }

  // Construct the image URL using the prompt and your preferred API
  const url = `http://localhost:8000/txt2img?prompt=${currentPrompt}`;

  // Fetch the image
  fetch(url)
    .then(response => response.blob())
    .then(blob => {
      // Create a URL for the blob image
      const imageUrl = URL.createObjectURL(blob);

      // Update the image container background
      imageContainer.style.backgroundImage = `url(${imageUrl})`;

      // Increment currentIndex and update image after timer
      if (currentIndex < totalPrompts - 1) {
        currentIndex++;
      } else {
        currentIndex = 0; // Reset to first prompt if all used
      }
      console.log(`Updated currentIndex: ${currentIndex}`);

      // Convert display time input to milliseconds for setTimeout
      const displayTime = displayTimeInput * 1000;

      // Start the timer directly within the callback
      imageTimerId = setTimeout(() => {
        console.log(`Inside timer func!`);
        // Release the used URL after display
        URL.revokeObjectURL(imageUrl);

        updateImage();
      }, displayTime);
    })
    .catch(error => console.error(error));
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
  const newPromptInput = document.createElement('input');
  const removePromptButton = document.createElement('button');
  removePromptButton.innerText = "X"

  // Set attributes and styles for the new elements
  newPromptInput.value = entryPromptInput.value; // Copy text from entry prompt
  entryPromptInput.value = ''; // Clear entry prompt after adding

  // Add functionality to remove button
  removePromptButton.addEventListener('click', () => {
    // Remove the parent element containing the input and button
    newPromptContainer.parentNode.removeChild(newPromptContainer);
  });

  // Append new elements to the container
  newPromptContainer.appendChild(newPromptInput);
  newPromptContainer.appendChild(removePromptButton);
  document.querySelector('#prompt-seq').appendChild(newPromptContainer);
});
