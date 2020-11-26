# Twitter Style Transfer webapp

[Demo link](https://calm-castle-91811.herokuapp.com/) - This link may take a while to load the first time. Heroku will shut down an apps servers if inactive for longer than 30 minutes, and it typically takes 20-30 seconds to reload.

## Webapp structure

This web app is made with a Flask backend and a React frontend. The backend is seen in the `app.py` file, which exposes an endpoint to be called from the frontend. The frontend code exists in the `client` subdirectory, with `client/src/App.jsx` containing the frontend logic and `client/src/App.scss` containing the styles.

## Running the backend

Ensure you are in the correct directory

```bash
cd TwitterStyleTransfer/webapp
```

Install project dependencies

```bash
pip install -r requirements.txt
```

Run the app

```bash
flask run
```

Visit the development site at http://127.0.0.1:5000/. Note that this will pull from the static `build/` directory, so any frontend changes will not be reflected here until the frontend is rebuit using the `npm run build` command (discussed in the frontend section).

## Running the frontend

Ensure you are in the correct directory

```bash
cd TwitterStyleTransfer/webapp/client
```

Install project dependencies

```bash
npm install
```

Run the app

```bash
npm start
```

You can now visit http://localhost:3000 and the page will be automatically reloaded when changes are made.

Build a optimized production code (necessary for frontend changes to be reflected in Heroku and when running flask)

```bash
npm run build
```

## Deploying to Heroku

Associate local folder with existing heroku deployment (you must be added to the heroku project)

```bash
cd TwitterStyleTransfer/webapp
heroku git:remote -a calm-castle-91811
```

As the Heroku deployment updates automatically when the related repository is pushed to, all we need to do is:

```bash
git add .
git commit -m "commit message"
git push heroku master
```

Changes should now be reflected on the [live link](https://calm-castle-91811.herokuapp.com/)
