MAX_RETRY_ATTEMPTS = 100    // retry 100 times
RETRY_DELAY = 1000          // one-second delay between attempts

function playVideo(video_element, video_url) {

    if (Hls.isSupported()) {

        let hls = new Hls({
            live: true,
            debug: false,
            manifestLoadPolicy: {
              default: {
                maxTimeToFirstByteMs: Infinity,
                maxLoadTimeMs: 10000,
                timeoutRetry: {
                  maxNumRetry: 60,
                  retryDelayMs: 1000,
                  maxRetryDelayMs: 1000,
                },
                errorRetry: {
                  maxNumRetry: 60,
                  retryDelayMs: 1000,
                  maxRetryDelayMs: 1000,
                },
              },
            },
        });
        hls.on(Hls.Events.MEDIA_ATTACHED, function () {
            console.log('media attached!');
            video_element.muted = true;
            video_element.play();
        });
        hls.on(Hls.Events.MANIFEST_PARSED, function () {
            console.log('manifest parsed!');
            hls.current?.play();
        });
        hls.on(Hls.Events.ERROR, function (event, data) {
            console.log(`HLS Error: ${event}: ${data}`);

            if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
                if (data.retry === undefined) {
                    data.retry = 1;
                }
                if (data.retry <= MAX_RETRY_ATTEMPTS) {
                    console.warn(`Manifest fetch failed (attempt ${data.retry}/${MAX_RETRY_ATTEMPTS}), retrying...`);
                    setTimeout(() => {
                        data.retry += 1;
                        hls.loadSource(video_url); // Retry loading the manifest
                    }, RETRY_DELAY);
                } else {
                    console.error(`Manifest fetch failed after ${MAX_RETRY_ATTEMPTS} attempts. Giving up.`);
                }
            }
        });
        hls.attachMedia(video_element);
        hls.loadSource(video_url);
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