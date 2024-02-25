def duplicate_frames(frames, target_fps):
    """Duplicates frames to achieve the target FPS.

    Args:
    blend: A list of blend.
    target_fps: The target frames per second.

    Returns:
    A list of blend with the target FPS achieved by duplicating frames.
    """

    # No need to calculate total_frames_needed, it's the target FPS itself
    total_frames_needed = target_fps

    # If there are enough blend, return them directly.
    if len(frames) >= total_frames_needed:
        return frames

    # Calculate how many times each image needs to be duplicated.
    duplicates_per_image, remainder = divmod(total_frames_needed, len(frames))

    # Create the output list.
    output_images = []

    # Add each image the required number of times.
    for img in frames:
        for _ in range(duplicates_per_image):
            output_images.append(img)

    # Add the remaining blend
    for _ in range(remainder):
        output_images.append(frames[-1])

    return output_images
