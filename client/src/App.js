import React, {Component} from "react";
import Keyboard from "./components/Keyboard";
import Results from "./components/Results";


class App extends Component {
    render() {
        return <div>
            <Results/>
            <Keyboard/>
        </div>
    }
}

export default App;