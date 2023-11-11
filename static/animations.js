document.addEventListener('DOMContentLoaded', function () {
    const videoElement = document.getElementById('selected-video');
    const animationSlider = document.getElementById('animation-slider');
    const animationSpeedSlider = document.getElementById('animation-speed');

    animationSlider.addEventListener('input', function () {
        const value = this.value;
        videoElement.playbackRate = value;
    });

    animationSpeedSlider.addEventListener('input', function () {
        const value = this.value;
        videoElement.style.animationDuration = `${value}s`;
    });
});
