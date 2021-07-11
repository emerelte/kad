import React, {Component} from "react";
import axios from "axios";
import "../style/Keyboard.css";
import TimePicker from "./TimePicker";
import Button from "@material-ui/core/Button";
import {TextField} from "@material-ui/core";

class Keyboard extends Component {
    constructor(props) {
        super(props);
        this.state = {
            metric: "",
            start_time: new Date("2014-04-01 14:00:00"),
            stop_time: new Date("2014-04-09 14:00:00"),
            message: ""
        };
        this.handleMetricChange = this.handleMetricChange.bind(this);
    }

    postUpdatedConfig = () => {
        axios.post("http://localhost:5000/update_config", {
            "METRIC_NAME": this.state.metric,
            "START_TIME": this.state.start_time.getTime() / 1000,
            "END_TIME": this.state.stop_time.getTime() / 1000
        }).then(response => {
            this.setState({message: "POST done!"});
        }).catch(() => {
            this.setState({message: "Error setting config!"})
        });
    }

    handleMetricChange(event) {
        this.setState({metric: event.target.value});
    }

    // TODO simplify
    handleStartDateChange = (date) => {
        this.setState({start_time: date});
        console.log(this.state.start_time)
    };

    handleStopDateChange = (date) => {
        this.setState({stop_time: date});
    };

    handleStartTimeChange = (date) => {
        this.state.start_time.setHours(date.getHours())
        this.state.start_time.setMinutes(date.getMinutes())
        this.state.start_time.setSeconds(date.getSeconds())
    };

    handleStopTimeChange = (date) => {
        this.state.stop_time.setHours(date.getHours())
        this.state.stop_time.setMinutes(date.getMinutes())
        this.state.stop_time.setSeconds(date.getSeconds())
    };

    render() {
        return <div className="keyboard">
            <form className="keyboard">
                <ul className="flex-outer">
                    <li>
                        <label>Metric</label>
                        <div style={{background: "lightgray"}}>
                            <TextField style={{width: "100%"}} type="text" value={this.state.metric}
                                       onChange={this.handleMetricChange}
                                       onKeyPress={(e) => {
                                           if (e.key === 'Enter') {
                                               this.postUpdatedConfig()
                                           }
                                       }}/>
                        </div>
                    </li>
                    <li>
                        <label>Start training time</label>
                        <TimePicker time={this.state.start_time} handleTimeChange={this.handleStartTimeChange}
                                    handleDateChange={this.handleStartDateChange}/>
                    </li>
                    <li>
                        <label>Stop training time</label>
                        <TimePicker time={this.state.stop_time} handleTimeChange={this.handleStopTimeChange}
                                    handleDateChange={this.handleStopDateChange}/>
                    </li>
                    <li>
                        <label/>
                        <Button variant="contained" color="secondary" onClick={this.postUpdatedConfig}>Submit
                            settings</Button>
                    </li>
                </ul>
            </form>
        </div>;
    }
}

export default Keyboard
