import React, { useState } from "react";
import "./App.scss";

// Import images
import DalaiLama from "./img/DalaiLama.jpg";
import dril from "./img/dril.jpg";
import elonmusk from "./img/elonmusk.jpg";
import realDonaldTrump from "./img/realDonaldTrump.jpg";

// Import Icons
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faInfoCircle } from "@fortawesome/free-solid-svg-icons";
import { faHeart, faComment } from "@fortawesome/free-regular-svg-icons";
import { faTwitter } from "@fortawesome/free-brands-svg-icons";

function App() {
  const accounts = {
    DalaiLama: { img: DalaiLama, name: "Dalai Lama" },
    dril: { img: dril, name: "wint" },
    elonmusk: { img: elonmusk, name: "Elon Musk" },
    realDonaldTrump: { img: realDonaldTrump, name: "Donald J. Trump" },
  };
  const defaultTweet = "Generated Tweet";
  const [tweet, setTweet] = useState(defaultTweet);
  const [text, setText] = useState("");
  const [account, setAccount] = useState(Object.keys(accounts)[0]);

  function styleTweet() {
    const apiUrl = "http://127.0.0.1:5000/api/style-tweet";

    // Build up the endpoint to our API
    const urlReqString = apiUrl + "?style=" + account + "&text=" + text;

    //Fetch the URL and parse the JSON response
    fetch(urlReqString)
      .then((res) => res.json())
      .then(
        (res) => {
          console.log(res);
          setTweet(res);
        },
        (error) => {
          console.log(error);
        }
      );
  }

  function getTwitterDate() {
    var months = [
      "January",
      "February",
      "March",
      "April",
      "May",
      "June",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December",
    ];
    const date = new Date();
    let hour = date.getHours();
    hour = hour % 12;
    hour = hour ? hour : 12; // the hour '0' should be '12'
    const amPm = hour <= 12 ? "AM" : "PM";
    let minutes = date.getMinutes();
    minutes = minutes < 10 ? "0" + minutes : minutes;
    const month = months[date.getMonth()];
    const day = date.getDate();
    const year = date.getFullYear();

    return (
      hour +
      ":" +
      minutes +
      " " +
      amPm +
      " · " +
      month +
      " " +
      day +
      ", " +
      year
    );
  }

  return (
    <>
      <div className="intro">
        <div className="intro-icon">
          <FontAwesomeIcon icon={faTwitter} color="rgba(29, 161, 242, .1)" />
        </div>
        <h1>
          <span>Twitter Style Transfer</span>
        </h1>
        <div className="about">
          <h2>How it works</h2>
          <p>
            Lorem ipsum dolor sit amet, consectetur adipisicing elit. Minima
            officiis cum minus, ullam fuga odio architecto saepe provident enim
            qui. Aut cupiditate, unde beatae rerum. Quos voluptas, harum animi
            suscipit.
          </p>
        </div>
      </div>

      <div className="compose">
        <h2>
          <span>Draft Tweet</span>
        </h2>

        <div className="dropdown">
          <select
            id="account"
            onChange={(e) => setAccount(e.target.value)}
            value={account}
          >
            {Object.keys(accounts).map((accountOption) => (
              <option value={accountOption}>@{accountOption}</option>
            ))}
          </select>
        </div>

        <div className="compose-tweet">
          <div className="flex-container">
            <img
              src={accounts[account].img}
              alt=""
              className="twit-profile-img"
            />
            <input
              type="text"
              name=""
              id="field"
              placeholder="What's happening?"
              onChange={(e) => setText(e.target.value)}
            />
          </div>

          <div className="share-button">
            <button className="twit-button" onClick={styleTweet}>
              Style Tweet
            </button>
          </div>
        </div>
      </div>

      {
        // Logic to not render tweet if the api has not been called yet
      }
      {tweet !== defaultTweet && (
        <div className="generated-content">
          <h2>Generated Tweet</h2>
          <div className="generated-tweet">
            <span className="twit-icon">
              <FontAwesomeIcon icon={faTwitter} color="#1D9BF0" />
            </span>

            <div className="profile">
              <img
                src={accounts[account].img}
                alt=""
                className="twit-profile-img"
              />
              <p className="name">{accounts[account].name} </p>
              <p className="light">@{account}</p>
            </div>

            <p>{tweet}</p>

            <div className="date">
              <p className="light">
                {getTwitterDate()}{" "}
                <span>
                  <FontAwesomeIcon icon={faInfoCircle} color="#666666" />
                </span>
              </p>
            </div>

            <div className="meta">
              <p className="light">
                <span className="like">
                  <FontAwesomeIcon icon={faHeart} color="#666666" /> 6k
                </span>
                <span className="comment">
                  <FontAwesomeIcon icon={faComment} color="#666666" /> 2.5k
                  people are tweeting about this
                </span>
              </p>
            </div>
          </div>

          <a
            href={"https://twitter.com/share?text=" + tweet}
            target="_blank"
            rel="noreferrer"
          >
            <button className="twit-button">Share this Tweet</button>
          </a>
        </div>
      )}

      <footer>Made in Austin &copy; 2020</footer>
    </>
  );
}

export default App;
