let ws_url = 'ws://localhost:8000/ws'

function connectWebSocket() {
  console.log('Connecting...');
  const ws = new WebSocket(ws_url);

  ws.onopen = () => {
    console.log(`WebSocket connection established! URL: ${ws_url}`);
  };

  ws.onerror = (error) => {
      console.error('WebSocket error:', error);
  };

  ws.onclose = () => {
      console.log('WebSocket connection closed. Initiating reconnection...');
      setTimeout(function() {
          connectWebSocket();
        }, 1000);
  };

  ws.onmessage = (event) => {
      console.log(`Incoming message: ${event.data}`)

      // Parse from JSON string
      const message = JSON.parse(event.data);

      // Check event type
      if (message.event === 'IMAGE_READY') {
          transitionImage(message.image_id);

      } else if (message.event === 'SEQUENCE_FINISHED') {
          console.log('Sequence Finished!')
      }
  };

  return ws;
}

connectWebSocket()
