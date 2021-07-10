import React, {Component} from "react";
import axios from "axios";
import {ComposedChart, Label, Legend, Line, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis} from "recharts";
import Button from "@material-ui/core/Button";
import "../style/Results.css"

const REFRESH_TIME_SEC = 1

class Results extends Component {
    constructor(props) {
        super(props);
        this.state = {
            message: null,
            image: null,
            rawData: null,
            visible: false
        };
    }

    componentDidMount() {
        // setInterval(() => {
        //     this.updateImage("http://localhost:5000/plot_results?timestamp=" + new Date().getTime());
        // }, REFRESH_TIME_SEC * 1000);
        setInterval(() => {
            this.updateData("http://localhost:5000/get_results?timestamp=" + new Date().getTime());
        }, REFRESH_TIME_SEC * 1000);
    }

    updateData(url) {
        axios.get(
            url,
            {responseType: "json"}
        ).then(response => {
            this.setState({rawData: response.data});
            this.setState({message: "Data updated"});
        }).catch(() => {
            this.setState({message: "Error fetching data!"})
            this.setState({rawData: null})
            this.setState({image: null});
            this.setState({visible: false});
        });
    }

    updateImage(url) {
        axios.get(
            url,
            {responseType: "arraybuffer"}
        )
            .then(response => {
                const base64 = btoa(
                    new Uint8Array(response.data).reduce(
                        (data, byte) => data + String.fromCharCode(byte),
                        "",
                    ),
                );
                this.setState({image: "data:;base64," + base64});
                this.setState({message: null});
            }).catch(() => {
            this.setState({message: "Error fetching image!"})
            this.setState({image: null});
        });
    }

    timeFromTimestamp = (timestamp) => {
        const date = new Date(timestamp * 1);

        const hours = "0" + date.getHours();
        const minutes = "0" + date.getMinutes();

        return hours.substr(-2) + ":" + minutes.substr(-2);
    }

    //TODO use formatter
    dateFromTimestamp = (timestamp) => {
        const date = new Date(timestamp * 1);

        const hours = "0" + date.getHours();
        const minutes = "0" + date.getMinutes();

        return date.getDate() + "." + ("0" + date.getMonth()).substr(-2) + "." + date.getFullYear() + ", " + hours.substr(-2) + ":" + minutes.substr(-2);
    }

    hideData = () => {
        this.setState({visible: false});
    }

    showData = () => {
        this.setState({visible: true});
    }

    render() {
        let metric_name = "";
        let data = [];

        const tooltipStyle = {
            fontWeight: "bold"
        }

        if (this.state.rawData) {
            console.log(this.state.rawData);
            metric_name = this.state.rawData["metric"];
            data = Object.entries(this.state.rawData[metric_name]).map(
                (e) => (
                    {
                        "raw_time": e[0],
                        "value": e[1],
                        "is_anomaly": this.state.rawData["is_anomaly"][e[0]] ? e[1] : null,
                        "predictions": this.state.rawData["predictions"][e[0]]
                    }))
            console.log(data);
        }
        if (!this.state.rawData) {
            return <Button disabled style={{color: "darkorange"}}>No data to show</Button>
        } else {
            return this.state.visible ?
                <div style={{position: "relative", width: "100%", height: 500}}><Button variant="contained"
                                                                                        onClick={this.hideData}>Hide
                    data</Button>
                    <ResponsiveContainer width="100%"
                                         height="100%">
                        <ComposedChart
                            data={data}
                            margin={{top: 15, right: 30, left: 20, bottom: 20}}>
                            <XAxis type="number" dataKey="raw_time" domain={["dataMin", "dataMax"]} tickCount={40}
                                   tickFormatter={this.timeFromTimestamp}>
                                <Label style={{fill: "white"}} value="time" position="bottom"/>
                            </XAxis>
                            <YAxis type="number" tickCount={10} domain={["auto", "auto"]}>
                                <Label style={{fill: "white"}} value={metric_name} angle={-90} position="left"/>
                            </YAxis>
                            <Tooltip itemStyle={tooltipStyle} labelFormatter={this.dateFromTimestamp}/>
                            <Line type="monotone" dataKey="value" stroke="orange" dot={false}/>
                            <Line type="monotone" dataKey="predictions" stroke="green" dot={false}/>
                            <Scatter dataKey="is_anomaly" fill="#5ABEF5FF" shape="diamond" legendType="diamond"/>
                            <Legend verticalAlign="top"/>
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>
                : <Button variant="contained" onClick={this.showData} style={{color: "green"}}>Show data</Button>
        }
    }
}

export default Results;