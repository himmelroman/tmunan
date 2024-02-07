function playVideo(video_url) {

    var video = document.getElementById('video');
    //
    // First check for native browser HLS support
    //
    if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = video_url;
    //
    // If no native HLS support, check if HLS.js is supported
    //
    }
    else if (Hls.isSupported()) {
        var hls = new Hls();
        hls.loadSource(video_url);
        hls.attachMedia(video);
    }
}
