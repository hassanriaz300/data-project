# Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).
Before npm start run fastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000


## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

The page will reload when you make changes.\
You may also see any lint errors in the console.

Not all the webapplication links are working left for future work.\

STEP BY STEP Running of Application.\

Clean service will clean the data for analysis .\
Input files should be in standard format i.e. name of columns.\
Edeka.xlsx is provide to user of app for standard.\
Output files after cleaning are saved as standard. edeka_reviews_wide.xlsx is provided to user of app.\

Facebook BART is used to further process and classify accusations(Seperately done in jypyter notebook because 14-15 hours processing time) .\
Output file -> edeka_reviews_zero_shot_accusations.xlsx .\

This will be input to :- .\
Semantic and keyword mapping done to further accurately classify 
top5_semantic_mappededeka.xlsx.\
top5_semantic_mappededeka.xlsx.\

Click on visualize or deep analysis and upload provided files e.g top5_semantic_mappededeka.xlsx .\

to analyse and visualize data.\

Different kind of plots can be generated, data set can be drilled down, city wise , branch wise or brand wise analysis can be done.\ 
Hotspot map is also available using google maps for drilling down further .\




### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
