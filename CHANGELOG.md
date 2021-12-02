# Changelog

## [1.3.0] - 2021-12-02

### Added
* Segmentation is now a standalone pip package #199 #209
* Automatic windows binary generation #171

### Changed
* Improved sample data handling
* Lots of dependencies

### Fixed
* Codecov now doesn't use depricated action #200


## [1.2.0] - 2021-08-27

### Added
* Kidney volume calculations #32
* Process multiple files at once #30 #194
* Menu bar with about, help and latest version links #40 #41

### Changed
* Post-processing is now off by default to make command line arguments more intuitive i.e. if you want to apply post-processing you add the `-p` flag

### Fixed
* CI badge
* Download links in readme
* Default GUI size (now doesn't start with a scroll bar)
* New releases aren't pre-releases by default any more


## [1.1.0] - 2021-08-24

### Added
* Tool now supports different fields of view #184 #190
* Post-processing option #183 #187
* Automatically run as CLI if arguments are passed #189 #191
* New release action #186

### Changed
* Weights are now stored on Zenodo for better visibility #185 #188
* Update readme

### Fixed
* Move CI to GitHub Actions #168
* Include link to paper and training data in readme


## [1.0.0] - 2020-10-06

### Changed
* File extension filters on input selection
* Lots of dependencies


## [0.1.0] - 2020-02-17

### Added
* Basic GUI
* Tests #6 #23 #35
* Travis CI #5

### Changed
* Lots of dependencies #11