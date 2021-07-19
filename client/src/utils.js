export const timeFromTimestamp = (timestamp) => {
    const date = new Date(timestamp * 1);

    const hours = "0" + date.getHours();
    const minutes = "0" + date.getMinutes();

    return hours.substr(-2) + ":" + minutes.substr(-2);
}

export const dateFromTimestamp = (timestamp) => {
    var dateFormat = require("dateformat");

    const date = new Date(timestamp * 1);

    return dateFormat(date, "yyyy-mm-dd HH:MM:ss");
}