import React, {Component} from "react";
import {KeyboardTimePicker, KeyboardDatePicker, MuiPickersUtilsProvider} from "@material-ui/pickers";
import DateFnsUtils from "@date-io/date-fns";
import Grid from "@material-ui/core/Grid";
import "../style/TimePicker.css"

class TimePicker extends Component {
    constructor(props) {
        super(props);
        this.state = {
            metric: "",
            message: ""
        };
    }

    render() {
        return <MuiPickersUtilsProvider utils={DateFnsUtils}>
            <Grid container justifyContent="space-around">
                <div className="date-picker" style={{background: "lightgray"}}>
                    <KeyboardDatePicker
                        margin="normal"
                        id="date-picker-dialog"
                        label="Select date"
                        format="dd/MM/yyyy"
                        value={this.props.time}
                        onChange={this.props.handleDateChange}
                        KeyboardButtonProps={{
                            'aria-label': 'change date',
                        }}
                    />
                    <KeyboardTimePicker
                        ampm={false}
                        margin="normal"
                        id="time-picker"
                        label="Select time"
                        value={this.props.time}
                        onChange={this.props.handleTimeChange}
                        KeyboardButtonProps={{
                            'aria-label': 'change time',
                        }}
                    />
                </div>
            </Grid>
        </MuiPickersUtilsProvider>
    }
}

export default TimePicker