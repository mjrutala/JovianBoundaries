<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
![GitHub contributors](https://img.shields.io/github/contributors/mjrutala/JovianBoundaries)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/mjrutala/JovianBoundaries)
![GitHub License](https://img.shields.io/github/license/mjrutala/JovianBoundaries)
[![Static Badge](https://img.shields.io/badge/arXiv-2502.09186-A42C25)](https://arxiv.org/abs/2502.09186)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14851327.svg)](https://doi.org/10.5281/zenodo.14851327)
<!--[![LinkedIn][linkedin-shield]][linkedin-url]-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/mjrutala/JovianBoundaries">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

<h3 align="center">JovianBoundaries</h3>

  <p align="center">
    A package containing magnetospheric boundary functions for Jupiter 
    <!-- <br />
    <a href="https://github.com/mjrutala/JovianBoundaries"><strong>Explore the docs »</strong></a>
    <br /> -->
    <br />
    <a href="#usage">Get Started</a>
    &middot;
    <a href="https://github.com/mjrutala/JovianBoundaries/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/mjrutala/JovianBoundaries/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a>
    <li><a href="#usage">Usage</a></li>
    <li>
      <a href="#quick-start">Quick Start</a>
      <a href="#documentation">Documentation</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

This repo contains the Bayesian model used to estimate Jupiter's magnetospheric boundaries from scarce, in-situ spacecraft data with imperfect coverage over the planet.

The repo is currently available as-is; future releases will include convenience functions for locating and plotting Jupiter's magneotpause and bow shock boundary surfaces.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->
<!-- ## Usage 

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->

<!-- QUICK START -->
## Quick Start

The function contained in `code/find_Boundary.py` can be used to estimate the locations of Jupiter's magnetospheric boundaries: the spatial coordinates or solar wind dynamic pressure may be estimated, with uncertainties, by providing enough information.

This repo is note packaged: it cannot be installed using `pip install ...` or `conda install...`, nor can it be imported using `import ...`. Instead, after downloading the repo, add the lines:
```python
import sys
sys.path.append("/full/path/to/JovianBoundaries/code/")
from find_Boundary import find_Boundary
```
in the import section of your code to use the `find_Boundary` function.

Some examples of `find_Boundary` are available in [this Jupyter Notebook](https://github.com/mjrutala/JovianBoundaries/blob/main/code/JovianBoundariesExamples.ipynb).

<!-- The boundary model definitions are available in `code/BoundaryModels.py`; at present, the coefficients for these models need to be obtained from Rutala et al. (2025, submitted). -->

## Documentation

### Prerequisites

The `find_Boundary` function is dependent only on basic python packages (e.g. `numpy`), which are easily obtainable. 
The code required to MCMC sample the posterior of the Bayesian boundary model has additional dependencies. The easiest way to get started with this code is create a new conda environment with all the dependencies. To do this, navigate to the directory containing this package and run:
```
>>conda env create -f environment.yml
>>conda activate huxt
```

<!-- ### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/mjrutala/JovianBoundaries.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
5. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin mjrutala/JovianBoundaries
   git remote -v # confirm the changes
   ``` -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
<!-- ## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/mjrutala/JovianBoundaries/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTRIBUTING -->
<!-- ## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->

### Top contributors:

<a href="https://github.com/mjrutala/JovianBoundaries/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mjrutala/JovianBoundaries" alt="contrib.rocks image" />
</a>


<!-- LICENSE -->
## License

Distributed under the MIT license. See `license.md` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Matthew J. Rutala - mrutala@cp.dias.ie

Project Link: [https://github.com/mjrutala/JovianBoundaries](https://github.com/mjrutala/JovianBoundaries)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
<!-- ## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/mjrutala/JovianBoundaries.svg?style=for-the-badge
[contributors-url]: https://github.com/mjrutala/JovianBoundaries/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/mjrutala/JovianBoundaries.svg?style=for-the-badge
[forks-url]: https://github.com/mjrutala/JovianBoundaries/network/members
[stars-shield]: https://img.shields.io/github/stars/mjrutala/JovianBoundaries.svg?style=for-the-badge
[stars-url]: https://github.com/mjrutala/JovianBoundaries/stargazers
[issues-shield]: https://img.shields.io/github/issues/mjrutala/JovianBoundaries.svg?style=for-the-badge
[issues-url]: https://github.com/mjrutala/JovianBoundaries/issues
[license-shield]: https://img.shields.io/github/license/mjrutala/JovianBoundaries.svg?style=for-the-badge
[license-url]: https://github.com/mjrutala/JovianBoundaries/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
