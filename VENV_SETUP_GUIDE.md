# Brain Tumor Detection - Virtual Environment Setup Guide

## ✅ Environment Setup Complete!

Your virtual environment has been successfully created and the application is now running.

## 🚀 How to Use Your Environment

### 1. **Activate the Virtual Environment**
Every time you want to work on this project, you need to activate the virtual environment:

**On Windows (PowerShell):**
```powershell
.\venv\Scripts\activate
```

**On Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
source venv/bin/activate
```

You'll know it's activated when you see `(venv)` at the beginning of your command prompt.

### 2. **Run the Application**
Once the virtual environment is activated, run:
```bash
python main.py
```

### 3. **Access the Web Application**
Open your web browser and go to:
```
http://localhost:5000
```

## 📁 Project Structure
- `main.py` - The main Flask application
- `models/model.h5` - The trained brain tumor detection model
- `templates/` - HTML templates for the web interface
- `uploads/` - Folder where uploaded images are stored
- `requirements_fixed.txt` - Fixed dependencies list

## 🔧 Troubleshooting

### If you get "Module not found" errors:
1. Make sure the virtual environment is activated (you should see `(venv)` in your prompt)
2. Reinstall dependencies: `pip install -r requirements_fixed.txt`

### If the model file is missing:
Make sure the `models/model.h5` file exists in your project directory.

### To deactivate the virtual environment:
```bash
deactivate
```

## 🎯 What This Application Does
This is a Flask web application that uses a trained TensorFlow model to detect brain tumors in MRI images. It can classify images into:
- Pituitary tumor
- Glioma tumor  
- Meningioma tumor
- No tumor

## 📝 Quick Commands Reference
```bash
# Activate environment
.\venv\Scripts\activate

# Run application
python main.py

# Deactivate environment
deactivate
```

Your application should now be running at http://localhost:5000! 🎉


