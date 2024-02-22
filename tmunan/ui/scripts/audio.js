function sendAudio(ws_url) {

    const ws = initWebsocket(ws_url)

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            console.log('Audio: Creating RecordRTC...');
            const recorder = new RecordRTC(stream, {
                type: 'audio',
                mimeType: 'audio/wav',
                timeSlice: 5000,    // Record in X second chunks
                recorderType: StereoAudioRecorder,
                numberOfAudioChannels: 2,
                desiredSampRate: 16000,
                ondataavailable: event => {
                    // console.log(`Audio: Sending data!`);
                    ws.send(event);
                },
                onStop: () => {
                    console.log('Audio: Recording stopped');
                }
            });

            recorder.startRecording();
            console.log('Audio: Started recording!');
      });
}

function initWebsocket(ws_url) {

    console.log('Audio: Connecting...');
    const ws = new WebSocket(ws_url);

    ws.onopen = () => {
        console.log(`Audio: WebSocket connection established! URL: ${ws_url}`);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
        console.log('WebSocket connection closed. Initiating reconnection...');
        // setTimeout(function() {
        //     connectWebSocket();
        // }, 1000);
    };

    ws.onmessage = (event) => {

        // Parse from JSON string
        const message = JSON.parse(event.data);
        console.log(`WS Message: ${message}`)
    };

  return ws;
}
