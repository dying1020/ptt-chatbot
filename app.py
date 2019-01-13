from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from model import get_model, get_bot_message
import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
socketio = SocketIO(app)

modelpath = 'model/best.tar'
encoder, decoder, searcher, dct = get_model(modelpath)

@app.route('/')
def sessions():
    return render_template('chat.html')

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print('received my event: ' + str(json))

    if json['user_message']:
        if len(json['user_name']) == 0:
            json['user_name'] = '肥宅'
        json['bot_name'] = '鄉民'
        json['bot_message'] = get_bot_message(json['user_message'], encoder, decoder, searcher, dct)
        now = datetime.datetime.now()
        json['time'] = now.strftime('%Y{y}%m{m}%d{d} %H:%M').format(y='年', m='月', d='日')
        print(str(json))
        emit('my response', json, callback=messageReceived, broadcast=True)
    else:
        print('Ignore event due to empty message.')

if __name__ == '__main__':
    socketio.run(app.run(host='0.0.0.0', debug=True))