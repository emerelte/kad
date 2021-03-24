import React, {Component} from 'react';
import axios from 'axios';

class App extends Component {
    state = {
        message: null,
        source: null
    };

    componentDidMount() {
        setInterval(() => {
            this.setImage("http://localhost:5000/plot_results?timestamp=" + new Date().getTime());
            this.updateData("http://localhost:5000/update_data");
        }, 10000);
    }

    updateData = (url) => {
        fetch(url)
            .then((response) => {
                if (response.status === 201) {
                    this.setState({message: "Data updated!"});
                } else {
                    throw Error(response.statusText)
                }
            }).catch(error => {
            this.setState({message: "Error updating data " + error});
        })
    }

    setImage(url) {
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
            }).catch((err) => {
                this.setState({message: "Error fetching image!"})
                this.setState({source: null});
                console.error(err)
        });
    }

    render() {
        return this.state.message === null ? <img src={this.state.source}/> : <h>{this.state.message}</h>;
    }
}

export default App;