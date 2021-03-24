import React, {Component} from 'react';
import axios from 'axios';

const REFRESH_TIME_SEC = 1

class App extends Component {
    state = {
        message: null,
        source: null
    };

    componentDidMount() {
        setInterval(() => {
            this.updateImage("http://localhost:5000/plot_results?timestamp=" + new Date().getTime());
        }, REFRESH_TIME_SEC * 1000);
    }

    updateImage(url) {
        axios.get(
            url,
            {responseType: 'arraybuffer'}
        )
            .then(response => {
                const base64 = btoa(
                    new Uint8Array(response.data).reduce(
                        (data, byte) => data + String.fromCharCode(byte),
                        '',
                    ),
                );
                this.setState({source: "data:;base64," + base64});
                this.setState({message: null});
            }).catch(() => {
            this.setState({message: "Error fetching image!"})
            this.setState({source: null});
        });
    }

    render() {
        return this.state.message === null ? <img src={this.state.source}/> : <h>{this.state.message}</h>;
    }
}

export default App;