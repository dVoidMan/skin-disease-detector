// DermAI - Main Client-Side Interactivity

document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const imageInput = document.getElementById('image-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewContainer = document.getElementById('preview-container');
    const dropzoneEmpty = document.getElementById('dropzone-empty');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const fileInfo = document.getElementById('file-info');
    const submitBtn = document.getElementById('submit-btn');
    const predictionForm = document.getElementById('prediction-form');

    const btnText = document.getElementById('btn-text');
    const btnSpinner = document.getElementById('btn-spinner');
    const btnLoadingText = document.getElementById('btn-loading-text');

    if (!dropzone || !imageInput) return;

    // Trigger file chooser when Browse button is clicked
    if (browseBtn) {
        browseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            imageInput.click();
        });
    }

    // Click anywhere on empty dropzone to open chooser
    dropzone.addEventListener('click', (e) => {
        if (e.target !== removeBtn && !previewContainer.contains(e.target)) {
            imageInput.click();
        }
    });

    // Drag-and-drop event listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    // Handle dropped files
    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // Handle file chosen through file dialogue
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Validation & Preview Handler
    function handleFileSelect(file) {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
        const maxSize = 16 * 1024 * 1024; // 16MB

        if (!allowedTypes.includes(file.type)) {
            alert('Please select a valid image file (JPEG, PNG, or WEBP).');
            return;
        }

        if (file.size > maxSize) {
            alert('File exceeds the maximum 16MB size limit.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzoneEmpty.classList.add('d-none');
            previewContainer.classList.remove('d-none');
            
            const sizeKB = (file.size / 1024).toFixed(1);
            fileInfo.textContent = `${file.name} (${sizeKB} KB)`;

            if (submitBtn) {
                submitBtn.removeAttribute('disabled');
            }
        };
        reader.readAsDataURL(file);
    }

    // Remove Selected Image
    if (removeBtn) {
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            imageInput.value = '';
            imagePreview.src = '#';
            previewContainer.classList.add('d-none');
            dropzoneEmpty.classList.remove('d-none');
            if (submitBtn) {
                submitBtn.setAttribute('disabled', 'true');
            }
        });
    }

    // Form Submission UI Progress Feedback
    if (predictionForm) {
        predictionForm.addEventListener('submit', () => {
            if (submitBtn) {
                submitBtn.setAttribute('disabled', 'true');
            }
            if (btnText) btnText.classList.add('d-none');
            if (btnSpinner) btnSpinner.classList.remove('d-none');
            if (btnLoadingText) btnLoadingText.classList.remove('d-none');
        });
    }
});
