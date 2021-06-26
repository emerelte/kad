import React, {Component} from "react";
import axios from "axios";
import {
    Line,
    Tooltip,
    Legend,
    XAxis,
    YAxis,
    ComposedChart, Scatter,
    Label
} from "recharts";

const REFRESH_TIME_SEC = 1

class App extends Component {
    state = {
        message: null,
        source: null,
        rawData: null
    };

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
            this.setState({source: null});
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
                this.setState({source: "data:;base64," + base64});
                this.setState({message: null});
            }).catch(() => {
            this.setState({message: "Error fetching image!"})
            this.setState({source: null});
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

    render() {
        let data = [];
        if (this.state.rawData) {
            console.log(this.state.rawData);
            data = Object.entries(this.state.rawData["value"]).map(
                (e) => (
                    {
                        "raw_time": e[0],
                        "value": e[1],
                        "is_anomaly": this.state.rawData["is_anomaly"][e[0]] ? e[1] : null,
                        "predictions": this.state.rawData["predictions"][e[0]]
                    }))
            console.log(data);
        }
        return this.state.rawData ?
            <ComposedChart
                width={1000}
                height={400}
                data={data}
                margin={{ top: 15, right: 30, left: 20, bottom: 20 }}
            >
                <XAxis type="number" dataKey="raw_time" domain={["dataMin", "dataMax"]} tickCount={40}
                       tickFormatter={this.timeFromTimestamp}>
                    <Label value="time" position="bottom" />
                </XAxis>
                <YAxis label={{ value: "metric", angle: -90, position: "left" }} type="number" tickCount={10} domain={["auto", "auto"]}/>
                <Tooltip payload={[{"xd": "xd"}]} viewBox={{x: 0, y: 0, width: 40, height: 40}}
                         labelFormatter={this.dateFromTimestamp}/>
                <Line type="monotone" dataKey="value" stroke="#ff7300" dot={false}/>
                <Line type="monotone" dataKey="predictions" stroke="#82ca9d" dot={false}/>
                <Scatter dataKey="is_anomaly" fill="blue" shape="diamond" legendType="diamond"/>
                <Legend verticalAlign="top"/>
            </ComposedChart>
            :
            <h>{this.state.message}</h>
    }
}

export default App;