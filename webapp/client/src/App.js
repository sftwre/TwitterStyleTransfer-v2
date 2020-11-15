import React, { useState } from "react";
import './App.scss';

// Import images
import DalaiLama from "./img/DalaiLama.jpg";
import dril from "./img/dril.jpg";
import elonmusk from "./img/elonmusk.jpg";
import realDonaldTrump from "./img/realDonaldTrump.jpg";


function App() {
  const accounts = ["DalaiLama","dril","elonmusk","realDonaldTrump"];
  const [tweet, setTweet] = useState("");
  const [text, setText] = useState("");
  const [account, setAccount] = useState(accounts[0]);

  function styleTweet() {
    const apiUrl = "http://127.0.0.1:5000/api/style-tweet";

      // Build up the endpoint to our API
     const urlReqString = apiUrl+ "?style=" + account + "&text=" + text;

     //Fetch the URL and parse the JSON response
     fetch(urlReqString)
       .then(res => res.json())
       .then(res => {
          console.log(res)
          setTweet(res);
       },
       (error) => {
         console.log(error);
       }
      );
  };

  return (
    <>
      <h1>Twitter Style Transfer</h1>
      <div className="about">
        <h2>How it works</h2>
        <p>Lorem ipsum dolor sit amet, consectetur adipisicing elit. Minima officiis cum minus, ullam fuga odio architecto saepe provident enim qui. Aut cupiditate, unde beatae rerum. Quos voluptas, harum animi suscipit.</p>
      </div>



      <div className="compose-tweet">
        <img src="" alt="" className="pfp"/>
        <h3>@{account}</h3>
        <select id="account" onChange={e => setAccount(e.target.value)} value={account}>
          {accounts.map(account => (
            <option value={account}>@{account}</option>
          ))}
        </select>
        <textarea name="" id="compose" cols="30" rows="10" onChange={(e) => setText(e.target.value)}></textarea>
        <button onClick={styleTweet}>Style Tweet</button>
      </div>

      <div className="generated-tweet">
        <img src="" alt="" className="pfp"/>
        <p>Generated tweet: {tweet}</p>
      </div>

      <button>Share this Tweet</button>

      <footer>
        <p>Made in Austin &copy; 2020</p>
      </footer>
    </>
  );
}

export default App;
