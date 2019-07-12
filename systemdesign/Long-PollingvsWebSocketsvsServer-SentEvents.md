# Ajax Polling

client 자주 request 한다. server 는 바로 response 한다.

# HTTP Long-Polling

client request 하고 오랫동안 기다린다. server 는 새로운 데이터가 있으면 response 한다.

# WebSockets

client HTTP 를 이용하여 connection 을 생성하고 HTTP 통신한다.

* client

```js
$(function () {
  // if user is running mozilla then use it's built-in WebSocket
  window.WebSocket = window.WebSocket || window.MozWebSocket;

  const connection = new WebSocket('ws://localhost:8080/githubEvents');

  connection.onopen = function () {
    // connection is opened and ready to use
  };

  connection.onerror = function (error) {
    // an error occurred when sending/receiving data
  };

  connection.onmessage = function (message) {
    // try to decode json (I assume that each message
    // from server is json)
    try {
      const githubEvent = JSON.parse(message.data); // display to the user appropriately
    } catch (e) {
      console.log('This doesn\'t look like a valid JSON: '+ message.data);
      return;
    }
    // handle incoming message
  };
});
```

* server

```js
const express = require('express');
const events = require('./events');
const path = require('path');

const app = express();

const port = process.env.PORT || 5001;

const expressWs = require('express-ws')(app);

app.get('/', function(req, res) {
	res.sendFile(path.join(__dirname + '/static/index.html'));
});

app.ws('/', function(ws, req) {
	const githubEvent = {}; // sample github Event from Github event API https://api.github.com/events
	ws.send('message', githubEvent);
});

app.listen(port, function() {
	console.log('Listening on', port);
});
```

# Server-Sent Events (SSEs)

서버에서 클라이언트로 데이터를 푸시할 수 있는 기술이다. HTML5 에 표준화 되어 있다.

* [Node.js - SSE (Server-Sent Events) Example + Javascript Client](https://www.woolha.com/tutorials/node-js-sse-server-sent-events-example-javascript-client)
* [SSE를 이용한 실시간 웹앱 @ spoqa](https://spoqa.github.io/2014/01/20/sse.html)

* client

```js
 const evtSource = new EventSource('/events');

 evtSource.addEventListener('event', function(evt) {
      const data = JSON.parse(evt.data);
      // Use data here
 },false);
```

* server

```js
// events.js
const EventEmitter = require('eventemitter3');
const emitter = new EventEmitter();

function subscribe(req, res) {

	res.writeHead(200, {
		'Content-Type': 'text/event-stream',
		'Cache-Control': 'no-cache',
		Connection: 'keep-alive'
	});

	// Heartbeat
	const nln = function() {
		res.write('\n');
	};
	const hbt = setInterval(nln, 15000);

	const onEvent = function(data) {
		res.write('retry: 500\n');
		res.write(`event: event\n`);
		res.write(`data: ${JSON.stringify(data)}\n\n`);
	};

	emitter.on('event', onEvent);

	// Clear heartbeat and listener
	req.on('close', function() {
		clearInterval(hbt);
		emitter.removeListener('event', onEvent);
	});
}

function publish(eventData) {
  // Emit events here recieved from Github/Twitter APIs
	emitter.emit('event', eventData);
}

module.exports = {
	subscribe, // Sending event data to the clients 
	publish // Emiting events from streaming servers
};

// App.js
const express = require('express');
const events = require('./events');
const port = process.env.PORT || 5001;
const app = express();

app.get('/events', cors(), events.subscribe);

app.listen(port, function() {
   console.log('Listening on', port);
});
```



# References

* [Polling vs SSE vs WebSocket— How to choose the right one](https://codeburst.io/polling-vs-sse-vs-websocket-how-to-choose-the-right-one-1859e4e13bd9)