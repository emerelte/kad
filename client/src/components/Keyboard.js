import React, {Component} from "react";
import axios from "axios";
import "../style/Keyboard.css";
import TimePicker from "./TimePicker";
import Button from "@material-ui/core/Button";
import {TextField} from "@material-ui/core";
import {dateFromTimestamp} from "../utils";
import Autocomplete from '@material-ui/lab/Autocomplete';

const availableModels = [
    "AutoEncoderModel",
    "HmmModel",
    "LstmModel",
    "SarimaModel"]

class Keyboard extends Component {
    constructor(props) {
        super(props);
        this.state = {
            metric: "",
            model: "",
            start_time: new Date("2014-04-01 14:00:00"),
            stop_time: new Date("2014-04-09 14:00:00"),
            message: ""
        };
        this.handleMetricChange = this.handleMetricChange.bind(this);
    }

    postUpdatedConfig = () => {
        if (this.state.metric == null || this.state.start_time == null || this.state.stop_time == null) {
            return;
        }
        axios.post("http://localhost:5000/update_config", {
            "METRIC_NAME": this.state.metric,
            "MODEL_NAME": this.state.model,
            "START_TIME": dateFromTimestamp(this.state.start_time.getTime()),
            "END_TIME": dateFromTimestamp(this.state.stop_time.getTime())
        }).then(response => {
            this.setState({message: "POST done!"});
        }).catch(() => {
            this.setState({message: "Error setting config!"})
        });
    }

    handleMetricChange(event) {
        this.setState({metric: event.target.value});
    }

    handleModelChange = (event) => {
        this.setState({model: event.target.textContent});
    }

    handleStartDateChange = (date) => {
        this.setState({start_time: date});
    };

    handleStopDateChange = (date) => {
        this.setState({stop_time: date});
    };

    handleStartTimeChange = (date) => {
        if (date != null) {

            this.state.start_time.setHours(date.getHours())
            this.state.start_time.setMinutes(date.getMinutes())
            this.state.start_time.setSeconds(date.getSeconds())
        }
    };

    handleStopTimeChange = (date) => {
        if (date != null) {
            this.state.stop_time.setHours(date.getHours())
            this.state.stop_time.setMinutes(date.getMinutes())
            this.state.stop_time.setSeconds(date.getSeconds())
        }
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
                        <label>Model</label>
                        <div style={{background: "lightgray", padding: "1%"}}>
                            <Autocomplete
                                id="model-combo-box"
                                options={availableModels}
                                label="Aut"
                                style={{width: "100%", marginTop: "1%"}}
                                onChange={this.handleModelChange}
                                renderInput={(params) => this.state.model ?
                                    <TextField {...params} label="Manual model selection"
                                               variant="outlined"/> :
                                    <TextField {...params} label="Automatic model selection"
                                               variant="outlined"/>}
                            /></div>
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
