import React, {Component} from "react";
import axios from "axios";

class Keyboard extends Component {
    constructor(props) {
        super(props);
        this.state = {metric: "", message: ""};
        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }

    updateKadConfig() {
        axios.post("http://localhost:5000/update_config", {"metric:": this.state.metric})
            .then(response => {
                this.setState({message: "POST done!"});
            }).catch(() => {
            this.setState({message: "Error setting config!"})
        });
    }

    handleChange(event) {
        this.setState({metric: event.target.value});
    }

    handleSubmit(event) {
        // alert("Zaktualizowano metrykę: " + this.state.metric);
        this.updateKadConfig();
        event.preventDefault();
    }

    render() {
        return <div style={{color: "white"}}>
            <h1>Message: {this.state.message}</h1>
            <form onSubmit={this.handleSubmit}><label>
                <input type="submit" value="Submit settings"/>
                Metric:
                <input type="text" value={this.state.metric} onChange={this.handleChange}/> </label>
            </form>
        </div>;
    }
}

export default Keyboard