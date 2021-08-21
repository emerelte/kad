import React, {Component} from "react";
import axios from "axios";
import {
    ComposedChart,
    Label,
    Legend,
    Line,
    ResponsiveContainer,
    Scatter,
    Tooltip as RechartsTooltip,
    XAxis,
    YAxis
} from "recharts";
import Button from "@material-ui/core/Button";
import Tooltip from '@material-ui/core/Tooltip';
import {dateFromTimestamp, timeFromTimestamp} from "../utils";
import "../style/Results.css";
import {withStyles} from "@material-ui/core";

const REFRESH_TIME_SEC = 2
const COLOR_BY_DATA = {"value": "orange", "predictions": "green", "is_anomaly": "#5ABEF5FF"}


const HtmlTooltip = withStyles((theme) => ({
    tooltip: {
        backgroundColor: '#f5f5f9',
        color: 'rgba(0, 0, 0, 0.87)',
        maxWidth: 220,
        fontSize: theme.typography.pxToRem(20),
        border: '1px solid #dadde9',
    },
}))(Tooltip);

const CustomRechartsTooltip = ({active, payload, label}) => {
        console.log(payload)
        if (active && payload && payload.length) {
            return (
                <div>
                    <h><b>{dateFromTimestamp(label)}</b></h>
                    <div>Value: <b style={{color: COLOR_BY_DATA["value"]}}>{payload[0].payload.value}</b></div>
                    {payload[0].payload.predictions ?
                        <div>
                            <div>Prediction: <b
                                style={{color: COLOR_BY_DATA["predictions"]}}>{payload[0].payload.predictions}</b></div>
                            <div>Is anomaly: <b
                                style={{color: COLOR_BY_DATA["is_anomaly"]}}>{payload[0].payload.is_anomaly ? "TRUE" : "FALSE"}</b>
                            </div>
                        </div>
                        : <div/>
                    }
                </div>
            );
        }

        return null;
    }
;

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

    hideGraph = () => {
        this.setState({visible: false});
    }

    showGraph = () => {
        this.setState({visible: true});
    }

    render() {
        let metric_name = "";
        let model = "";
        let data = [];

        const tooltipStyle = {
            fontWeight: "bold"
        }

        if (this.state.rawData) {
            console.log(this.state.rawData);
            metric_name = this.state.rawData["metric"];
            model = this.state.rawData["model"];
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
            return <div className="results-div-outer"><Button size="large" disabled
                                                              style={{color: "darkorange"}}>No
                data to show</Button></div>
        } else {
            return this.state.visible ?
                <div className="results-div-outer" style={{position: "relative", width: "100%", height: 500}}>
                    <ResponsiveContainer width="100%"
                                         height="100%">
                        <ComposedChart
                            data={data}
                            margin={{top: 15, right: 30, left: 20, bottom: 20}}>
                            <XAxis type="number" dataKey="raw_time" domain={["dataMin", "dataMax"]} tickCount={40}
                                   tickFormatter={timeFromTimestamp}>
                                <Label style={{fill: "white"}} value="time" position="bottom"/>
                            </XAxis>
                            <YAxis type="number" tickCount={10} domain={["auto", "auto"]}>
                                <Label style={{fill: "white"}} value={metric_name} angle={-90} position="left"/>
                            </YAxis>
                            <RechartsTooltip content={<CustomRechartsTooltip/>} itemStyle={tooltipStyle}/>
                            <Line type="monotone" dataKey="value" stroke={COLOR_BY_DATA["value"]} dot={false}/>
                            <Line type="monotone" dataKey="predictions" stroke={COLOR_BY_DATA["predictions"]}
                                  dot={false}/>
                            <Scatter dataKey="is_anomaly" fill={COLOR_BY_DATA["is_anomaly"]} shape="diamond"
                                     legendType="diamond"/>
                            <Legend verticalAlign="top"/>
                        </ComposedChart>
                    </ResponsiveContainer>
                    <div className="results-div-inner"><Button size="large" variant="contained"
                                                               onClick={this.hideGraph}>Hide graph</Button></div>
                </div>
                : <div className="results-div-outer">
                    <div className="results-div-inner"><HtmlTooltip placement="right" title={
                        <React.Fragment>
                            Show the results obtainted using: <b>{model}</b>
                        </React.Fragment>
                    }><Button variant="contained"
                              size="large"
                              onClick={this.showGraph}
                              style={{
                                  color: "green",
                                  width: "auto"
                              }}>Show
                        graph</Button></HtmlTooltip></div>
                </div>
        }
    }
}

export default Results;