import React, {Component} from "react";
import Keyboard from "./components/Keyboard";
import Results from "./components/Results";


class App extends Component {
    render() {
        return <div>
            <h1 style={{color: "lightgray", textAlign: "center"}}>Kubernetes Anomaly Detector</h1>
            <Results/>
            <Keyboard/>
        </div>
    }
}

export default App;