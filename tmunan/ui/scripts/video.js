function playVideo(video_element, video_url) {

    // var video = document.getElementById('video');
    if (Hls.isSupported()) {

        let hls = new Hls({
            debug: true,
        });

        hls.loadSource(video_url);
        hls.attachMedia(video_element);
        hls.on(Hls.Events.MEDIA_ATTACHED, function () {
            video_element.muted = true;
            video_element.play();
        });
    }

    // hls.js is not supported on platforms that do not have Media Source Extensions (MSE) enabled.
    // When the browser has built-in HLS support (check using `canPlayType`), we can provide an HLS manifest (i.e. .m3u8 URL) directly to the video element through the `src` property.
    // This is using the built-in support of the plain video element, without using hls.js.
    else if (video_element.canPlayType('application/vnd.apple.mpegurl')) {
        video_element.src = video_url;
        video_element.addEventListener('canplay', function () {
            video_element.play();
        });
    }
}