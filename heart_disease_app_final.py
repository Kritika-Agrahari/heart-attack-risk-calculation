#!/usr/bin/env python3
"""
Heart Disease Prediction System - Final Complete App
A comprehensive GUI application with data storage and professional interface
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import threading
from datetime import datetime
import json

class HeartDiseaseApp:
    def __init__(self):
        self.model = None
        self.data_file = 'heart_disease_data.csv'
        self.user_data_file = 'user_predictions.csv'
        self.setup_gui()
        self.load_model_async()
        
    def setup_gui(self):
        """Initialize the main GUI"""
        self.root = tk.Tk()
        self.root.title("Heart Disease Prediction System - Professional Edition")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f5f5f5')
        
        # Configure styles
        self.setup_styles()
        
        # Create main interface
        self.create_main_interface()
        
    def setup_styles(self):
        """Configure custom styles with colors and graphics"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Color scheme - Medical Blue Theme
        self.colors = {
            'primary_blue': '#2E86AB',
            'light_blue': '#A23B72',
            'accent_blue': '#F18F01',
            'success_green': '#C73E1D',
            'warning_orange': '#F18F01',
            'error_red': '#C73E1D',
            'background': '#F8FFFE',
            'card_bg': '#FFFFFF',
            'text_dark': '#2C3E50',
            'text_light': '#7F8C8D'
        }
        
        # Configure root background
        self.root.configure(bg=self.colors['background'])
        
        # Title styles with gradient effect
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 20, 'bold'), 
                       background=self.colors['background'], 
                       foreground=self.colors['primary_blue'])
        
        style.configure('Subtitle.TLabel', 
                       font=('Segoe UI', 12, 'bold'), 
                       background=self.colors['card_bg'], 
                       foreground=self.colors['text_dark'])
        
        style.configure('Info.TLabel', 
                       font=('Segoe UI', 10), 
                       background=self.colors['card_bg'], 
                       foreground=self.colors['text_light'])
        
        style.configure('Success.TLabel', 
                       font=('Segoe UI', 11, 'bold'), 
                       background=self.colors['background'], 
                       foreground=self.colors['success_green'])
        
        style.configure('Error.TLabel', 
                       font=('Segoe UI', 11, 'bold'), 
                       background=self.colors['background'], 
                       foreground=self.colors['error_red'])
        
        # Interactive button styles
        style.configure('Predict.TButton', 
                       font=('Segoe UI', 12, 'bold'), 
                       padding=(25, 12),
                       background=self.colors['primary_blue'],
                       foreground='white')
        
        style.map('Predict.TButton',
                 background=[('active', self.colors['light_blue']),
                           ('pressed', self.colors['accent_blue'])])
        
        style.configure('Action.TButton', 
                       font=('Segoe UI', 10, 'bold'), 
                       padding=(15, 8),
                       background=self.colors['accent_blue'],
                       foreground='white')
        
        style.map('Action.TButton',
                 background=[('active', self.colors['warning_orange']),
                           ('pressed', self.colors['primary_blue'])])
        
        # Card-like frame styles
        style.configure('Card.TLabelFrame', 
                       background=self.colors['card_bg'],
                       borderwidth=2,
                       relief='raised')
        
        style.configure('Card.TLabelFrame.Label', 
                       font=('Segoe UI', 12, 'bold'),
                       background=self.colors['card_bg'],
                       foreground=self.colors['primary_blue'])
        
    def create_main_interface(self):
        """Create the main application interface with colorful cards"""
        # Main container with gradient background
        main_frame = tk.Frame(self.root, bg=self.colors['background'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Create scrollable canvas for better UX
        canvas = tk.Canvas(main_frame, bg=self.colors['background'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['background'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Header section with animated elements
        self.create_animated_header(scrollable_frame)
        
        # Input section with card design
        self.create_card_input_section(scrollable_frame)
        
        # Interactive action buttons
        self.create_interactive_buttons(scrollable_frame)
        
        # Results section with graphics
        self.create_visual_results_section(scrollable_frame)
        
        # Data management with charts
        self.create_enhanced_data_section(scrollable_frame)
        
    def create_header(self, parent):
        """Create header section"""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Title
        title_label = ttk.Label(header_frame, text="🏥 Heart Disease Risk Assessment System", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Status
        self.status_label = ttk.Label(header_frame, text="🔄 Loading model...", 
                                     style='Info.TLabel')
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # Model info
        self.model_info_label = ttk.Label(header_frame, text="", style='Success.TLabel')
        self.model_info_label.grid(row=2, column=0, sticky=tk.W, pady=(2, 0))
        
    def create_input_section(self, parent):
        """Create input fields section"""
        # Input frame with border
        input_frame = ttk.LabelFrame(parent, text="📋 Patient Information", padding="15")
        input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        input_frame.columnconfigure(1, weight=1)
        input_frame.columnconfigure(3, weight=1)
        
        self.input_vars = {}
        
        # Feature definitions
        features = [
            ('age', 'Age (years)', 'Patient age', 1, 120, 0),
            ('sex', 'Sex', '0=Female, 1=Male', 0, 1, 1),
            ('cp', 'Chest Pain Type', '0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic', 0, 3, 2),
            ('trestbps', 'Resting BP (mmHg)', 'Resting blood pressure', 50, 300, 3),
            ('chol', 'Cholesterol (mg/dl)', 'Serum cholesterol level', 100, 600, 4),
            ('fbs', 'Fasting Blood Sugar', 'FBS > 120 mg/dl (0=No, 1=Yes)', 0, 1, 5),
            ('restecg', 'Resting ECG', '0=Normal, 1=ST-T abnormal, 2=LV hypertrophy', 0, 2, 6),
            ('thalach', 'Max Heart Rate', 'Maximum heart rate achieved', 50, 250, 7),
            ('exang', 'Exercise Angina', 'Exercise induced angina (0=No, 1=Yes)', 0, 1, 8),
            ('oldpeak', 'ST Depression', 'ST depression by exercise', 0.0, 10.0, 9),
            ('slope', 'ST Slope', '0=Upsloping, 1=Flat, 2=Downsloping', 0, 2, 10),
            ('ca', 'Major Vessels', 'Vessels colored by fluoroscopy (0-4)', 0, 4, 11),
            ('thal', 'Thalassemia', '0=Normal, 1=Fixed, 2=Reversible, 3=Not described', 0, 3, 12)
        ]
        
        # Create input fields in 2 columns
        for i, (feature, label, description, min_val, max_val, idx) in enumerate(features):
            row = idx // 2
            col = (idx % 2) * 2
            
            # Label
            ttk.Label(input_frame, text=f"{label}:", style='Subtitle.TLabel').grid(
                row=row*2, column=col, sticky=tk.W, padx=(0, 10), pady=2)
            
            # Input field
            if feature == 'oldpeak':
                self.input_vars[feature] = tk.DoubleVar(value=0.0)
                entry = ttk.Spinbox(input_frame, from_=min_val, to=max_val, increment=0.1,
                                   textvariable=self.input_vars[feature], width=15)
            else:
                self.input_vars[feature] = tk.IntVar(value=0)
                entry = ttk.Spinbox(input_frame, from_=min_val, to=max_val, increment=1,
                                   textvariable=self.input_vars[feature], width=15)
            
            entry.grid(row=row*2, column=col+1, sticky=tk.W, padx=(0, 20), pady=2)
            
            # Description
            ttk.Label(input_frame, text=description, style='Info.TLabel').grid(
                row=row*2+1, column=col, columnspan=2, sticky=tk.W, padx=(0, 20), pady=(0, 10))
        
    def create_action_buttons(self, parent):
        """Create action buttons"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        # Main predict button
        self.predict_button = ttk.Button(button_frame, text="🔍 Predict Heart Disease Risk", 
                                        command=self.predict_risk, style='Predict.TButton',
                                        state='disabled')
        self.predict_button.pack(side=tk.LEFT, padx=10)
        
        # Clear button
        clear_button = ttk.Button(button_frame, text="🗑️ Clear All Fields", 
                                 command=self.clear_fields, style='Action.TButton')
        clear_button.pack(side=tk.LEFT, padx=10)
        
        # Save prediction button
        self.save_button = ttk.Button(button_frame, text="💾 Save Prediction", 
                                     command=self.save_prediction, style='Action.TButton',
                                     state='disabled')
        self.save_button.pack(side=tk.LEFT, padx=10)
        
    def create_results_section(self, parent):
        """Create results display section"""
        results_frame = ttk.LabelFrame(parent, text="📊 Prediction Results", padding="15")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        results_frame.columnconfigure(0, weight=1)
        
        # Results display
        self.results_text = tk.Text(results_frame, height=12, width=80, font=('Consolas', 10),
                                   wrap=tk.WORD, bg='#ffffff', relief='sunken', bd=2)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Initial message
        self.show_initial_message()
        
    def create_data_section(self, parent):
        """Create data management section"""
        data_frame = ttk.LabelFrame(parent, text="📁 Data Management", padding="10")
        data_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Data info
        self.data_info_label = ttk.Label(data_frame, text="", style='Info.TLabel')
        self.data_info_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Data management buttons
        data_button_frame = ttk.Frame(data_frame)
        data_button_frame.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        ttk.Button(data_button_frame, text="📈 View Saved Predictions", 
                  command=self.view_saved_data, style='Action.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(data_button_frame, text="📤 Export Data", 
                  command=self.export_data, style='Action.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(data_button_frame, text="🔄 Retrain Model", 
                  command=self.retrain_model, style='Action.TButton').pack(side=tk.LEFT, padx=5)
    
    def create_animated_header(self, parent):
        """Create animated header with graphics"""
        header_card = tk.Frame(parent, bg=self.colors['card_bg'], relief='raised', bd=3)
        header_card.pack(fill=tk.X, pady=(0, 20), padx=10)
        
        # Title with medical icon
        title_frame = tk.Frame(header_card, bg=self.colors['card_bg'])
        title_frame.pack(fill=tk.X, pady=15)
        
        # Medical cross icon (created with text)
        icon_label = tk.Label(title_frame, text="⚕️", font=('Segoe UI', 24), 
                             bg=self.colors['card_bg'], fg=self.colors['primary_blue'])
        icon_label.pack(side=tk.LEFT, padx=(20, 10))
        
        # Title text
        title_label = tk.Label(title_frame, text="Heart Disease Risk Assessment System", 
                              font=('Segoe UI', 20, 'bold'), bg=self.colors['card_bg'], 
                              fg=self.colors['primary_blue'])
        title_label.pack(side=tk.LEFT, padx=10)
        
        # Status section with colored indicators
        status_frame = tk.Frame(header_card, bg=self.colors['card_bg'])
        status_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
        
        # Status indicator
        self.status_indicator = tk.Label(status_frame, text="●", font=('Arial', 16), 
                                        bg=self.colors['card_bg'], fg='orange')
        self.status_indicator.pack(side=tk.LEFT)
        
        self.status_label = tk.Label(status_frame, text="🔄 Loading model...", 
                                    font=('Segoe UI', 11), bg=self.colors['card_bg'], 
                                    fg=self.colors['text_dark'])
        self.status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Model info with icon
        self.model_info_label = tk.Label(status_frame, text="", 
                                        font=('Segoe UI', 10, 'bold'), 
                                        bg=self.colors['card_bg'], fg=self.colors['success_green'])
        self.model_info_label.pack(side=tk.RIGHT)
    
    def create_card_input_section(self, parent):
        """Create input section with card design"""
        input_card = tk.Frame(parent, bg=self.colors['card_bg'], relief='raised', bd=3)
        input_card.pack(fill=tk.X, pady=10, padx=10)
        
        # Card header
        header = tk.Frame(input_card, bg=self.colors['primary_blue'], height=40)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        header_label = tk.Label(header, text="📋 Patient Information", 
                               font=('Segoe UI', 14, 'bold'), bg=self.colors['primary_blue'], 
                               fg='white')
        header_label.pack(expand=True)
        
        # Input fields container
        fields_frame = tk.Frame(input_card, bg=self.colors['card_bg'])
        fields_frame.pack(fill=tk.X, padx=20, pady=20)
        
        self.input_vars = {}
        
        # Feature definitions with colors
        features = [
            ('age', 'Age (years)', 'Patient age', 1, 120, '#3498db'),
            ('sex', 'Sex', '0=Female, 1=Male', 0, 1, '#e91e63'),
            ('cp', 'Chest Pain Type', '0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic', 0, 3, '#ff9800'),
            ('trestbps', 'Resting BP (mmHg)', 'Resting blood pressure', 50, 300, '#f44336'),
            ('chol', 'Cholesterol (mg/dl)', 'Serum cholesterol level', 100, 600, '#9c27b0'),
            ('fbs', 'Fasting Blood Sugar', 'FBS > 120 mg/dl (0=No, 1=Yes)', 0, 1, '#4caf50'),
            ('restecg', 'Resting ECG', '0=Normal, 1=ST-T abnormal, 2=LV hypertrophy', 0, 2, '#00bcd4'),
            ('thalach', 'Max Heart Rate', 'Maximum heart rate achieved', 50, 250, '#ff5722'),
            ('exang', 'Exercise Angina', 'Exercise induced angina (0=No, 1=Yes)', 0, 1, '#795548'),
            ('oldpeak', 'ST Depression', 'ST depression by exercise', 0.0, 10.0, '#607d8b'),
            ('slope', 'ST Slope', '0=Upsloping, 1=Flat, 2=Downsloping', 0, 2, '#8bc34a'),
            ('ca', 'Major Vessels', 'Vessels colored by fluoroscopy (0-4)', 0, 4, '#ffc107'),
            ('thal', 'Thalassemia', '0=Normal, 1=Fixed, 2=Reversible, 3=Not described', 0, 3, '#673ab7')
        ]
        
        # Create input fields in a grid with colors
        for i, (feature, label, description, min_val, max_val, color) in enumerate(features):
            row = i // 2
            col = i % 2
            
            # Field container with colored border
            field_frame = tk.Frame(fields_frame, bg=self.colors['card_bg'], relief='solid', bd=1)
            field_frame.grid(row=row, column=col, sticky='ew', padx=10, pady=8)
            fields_frame.grid_columnconfigure(col, weight=1)
            
            # Colored indicator
            indicator = tk.Frame(field_frame, bg=color, width=4)
            indicator.pack(side=tk.LEFT, fill=tk.Y)
            
            # Content frame
            content_frame = tk.Frame(field_frame, bg=self.colors['card_bg'])
            content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=8)
            
            # Label with color
            label_widget = tk.Label(content_frame, text=f"{label}:", 
                                   font=('Segoe UI', 11, 'bold'), 
                                   bg=self.colors['card_bg'], fg=color)
            label_widget.pack(anchor='w')
            
            # Input field
            if feature == 'oldpeak':
                self.input_vars[feature] = tk.DoubleVar(value=0.0)
                entry = tk.Spinbox(content_frame, from_=min_val, to=max_val, increment=0.1,
                                  textvariable=self.input_vars[feature], width=15,
                                  font=('Segoe UI', 10), relief='solid', bd=1)
            else:
                self.input_vars[feature] = tk.IntVar(value=0)
                entry = tk.Spinbox(content_frame, from_=min_val, to=max_val, increment=1,
                                  textvariable=self.input_vars[feature], width=15,
                                  font=('Segoe UI', 10), relief='solid', bd=1)
            
            entry.pack(anchor='w', pady=2)
            
            # Description
            desc_label = tk.Label(content_frame, text=description, 
                                 font=('Segoe UI', 9), bg=self.colors['card_bg'], 
                                 fg=self.colors['text_light'], wraplength=200)
            desc_label.pack(anchor='w')
    
    def create_interactive_buttons(self, parent):
        """Create interactive action buttons with hover effects"""
        button_card = tk.Frame(parent, bg=self.colors['card_bg'], relief='raised', bd=3)
        button_card.pack(fill=tk.X, pady=10, padx=10)
        
        button_frame = tk.Frame(button_card, bg=self.colors['card_bg'])
        button_frame.pack(pady=20)
        
        # Main predict button with gradient effect
        self.predict_button = tk.Button(button_frame, text="🔍 Predict Heart Disease Risk", 
                                       command=self.predict_risk, 
                                       font=('Segoe UI', 12, 'bold'),
                                       bg=self.colors['primary_blue'], fg='white',
                                       relief='raised', bd=3, padx=30, pady=12,
                                       state='disabled', cursor='hand2')
        self.predict_button.pack(side=tk.LEFT, padx=15)
        
        # Bind hover effects
        self.predict_button.bind('<Enter>', lambda e: self.on_button_hover(e, self.colors['light_blue']))
        self.predict_button.bind('<Leave>', lambda e: self.on_button_leave(e, self.colors['primary_blue']))
        
        # Clear button
        clear_button = tk.Button(button_frame, text="🗑️ Clear All Fields", 
                                command=self.clear_fields,
                                font=('Segoe UI', 10, 'bold'),
                                bg=self.colors['accent_blue'], fg='white',
                                relief='raised', bd=2, padx=20, pady=8, cursor='hand2')
        clear_button.pack(side=tk.LEFT, padx=15)
        
        clear_button.bind('<Enter>', lambda e: self.on_button_hover(e, self.colors['warning_orange']))
        clear_button.bind('<Leave>', lambda e: self.on_button_leave(e, self.colors['accent_blue']))
        
        # Save button
        self.save_button = tk.Button(button_frame, text="💾 Save Prediction", 
                                    command=self.save_prediction,
                                    font=('Segoe UI', 10, 'bold'),
                                    bg=self.colors['success_green'], fg='white',
                                    relief='raised', bd=2, padx=20, pady=8,
                                    state='disabled', cursor='hand2')
        self.save_button.pack(side=tk.LEFT, padx=15)
        
        self.save_button.bind('<Enter>', lambda e: self.on_button_hover(e, '#27ae60'))
        self.save_button.bind('<Leave>', lambda e: self.on_button_leave(e, self.colors['success_green']))
    
    def on_button_hover(self, event, color):
        """Button hover effect"""
        event.widget.config(bg=color)
    
    def on_button_leave(self, event, color):
        """Button leave effect"""
        event.widget.config(bg=color)
    
    def create_visual_results_section(self, parent):
        """Create results section with visual elements"""
        results_card = tk.Frame(parent, bg=self.colors['card_bg'], relief='raised', bd=3)
        results_card.pack(fill=tk.X, pady=10, padx=10)
        
        # Card header with gradient
        header = tk.Frame(results_card, bg=self.colors['primary_blue'], height=40)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        header_label = tk.Label(header, text="📊 Assessment Results", 
                               font=('Segoe UI', 14, 'bold'), bg=self.colors['primary_blue'], 
                               fg='white')
        header_label.pack(expand=True)
        
        # Results container
        results_container = tk.Frame(results_card, bg=self.colors['card_bg'])
        results_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Results display with custom styling
        self.results_text = tk.Text(results_container, height=12, width=80, 
                                   font=('Consolas', 10), wrap=tk.WORD,
                                   bg='#fafafa', fg=self.colors['text_dark'],
                                   relief='solid', bd=2, padx=15, pady=15)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Custom scrollbar
        scrollbar = tk.Scrollbar(results_container, command=self.results_text.yview,
                                bg=self.colors['primary_blue'], troughcolor='#f0f0f0')
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure text tags for colored output
        self.results_text.tag_configure('header', font=('Segoe UI', 12, 'bold'), 
                                       foreground=self.colors['primary_blue'])
        self.results_text.tag_configure('success', font=('Segoe UI', 11, 'bold'), 
                                       foreground='#27ae60')
        self.results_text.tag_configure('warning', font=('Segoe UI', 11, 'bold'), 
                                       foreground='#f39c12')
        self.results_text.tag_configure('error', font=('Segoe UI', 11, 'bold'), 
                                       foreground='#e74c3c')
        self.results_text.tag_configure('info', font=('Segoe UI', 10), 
                                       foreground=self.colors['text_light'])
        
        # Initial styled message
        self.show_styled_initial_message()
    
    def create_enhanced_data_section(self, parent):
        """Create enhanced data management section with visual elements"""
        data_card = tk.Frame(parent, bg=self.colors['card_bg'], relief='raised', bd=3)
        data_card.pack(fill=tk.X, pady=10, padx=10)
        
        # Card header
        header = tk.Frame(data_card, bg=self.colors['accent_blue'], height=40)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        header_label = tk.Label(header, text="📁 Data Management & Analytics", 
                               font=('Segoe UI', 14, 'bold'), bg=self.colors['accent_blue'], 
                               fg='white')
        header_label.pack(expand=True)
        
        # Data info section
        info_frame = tk.Frame(data_card, bg=self.colors['card_bg'])
        info_frame.pack(fill=tk.X, padx=20, pady=15)
        
        # Data statistics with icons
        stats_frame = tk.Frame(info_frame, bg=self.colors['card_bg'])
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Saved predictions counter
        self.saved_count_frame = tk.Frame(stats_frame, bg='#ecf0f1', relief='solid', bd=1)
        self.saved_count_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        tk.Label(self.saved_count_frame, text="💾", font=('Arial', 16), 
                bg='#ecf0f1').pack(side=tk.LEFT, padx=5)
        self.data_info_label = tk.Label(self.saved_count_frame, text="No saved predictions yet", 
                                       font=('Segoe UI', 10, 'bold'), bg='#ecf0f1', 
                                       fg=self.colors['text_dark'])
        self.data_info_label.pack(side=tk.LEFT, padx=5)
        
        # Model accuracy display
        self.accuracy_frame = tk.Frame(stats_frame, bg='#e8f5e8', relief='solid', bd=1)
        self.accuracy_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        tk.Label(self.accuracy_frame, text="🎯", font=('Arial', 16), 
                bg='#e8f5e8').pack(side=tk.LEFT, padx=5)
        self.accuracy_label = tk.Label(self.accuracy_frame, text="Model Loading...", 
                                      font=('Segoe UI', 10, 'bold'), bg='#e8f5e8', 
                                      fg=self.colors['success_green'])
        self.accuracy_label.pack(side=tk.LEFT, padx=5)
        
        # Action buttons with enhanced styling
        button_frame = tk.Frame(info_frame, bg=self.colors['card_bg'])
        button_frame.pack(fill=tk.X, pady=15)
        
        # View data button
        view_btn = tk.Button(button_frame, text="📈 View Saved Predictions", 
                            command=self.view_saved_data,
                            font=('Segoe UI', 10, 'bold'),
                            bg='#3498db', fg='white', relief='raised', bd=2,
                            padx=15, pady=8, cursor='hand2')
        view_btn.pack(side=tk.LEFT, padx=5)
        view_btn.bind('<Enter>', lambda e: self.on_button_hover(e, '#2980b9'))
        view_btn.bind('<Leave>', lambda e: self.on_button_leave(e, '#3498db'))
        
        # Export button
        export_btn = tk.Button(button_frame, text="📤 Export Data", 
                              command=self.export_data,
                              font=('Segoe UI', 10, 'bold'),
                              bg='#9b59b6', fg='white', relief='raised', bd=2,
                              padx=15, pady=8, cursor='hand2')
        export_btn.pack(side=tk.LEFT, padx=5)
        export_btn.bind('<Enter>', lambda e: self.on_button_hover(e, '#8e44ad'))
        export_btn.bind('<Leave>', lambda e: self.on_button_leave(e, '#9b59b6'))
        
        # Retrain button
        retrain_btn = tk.Button(button_frame, text="🔄 Retrain Model", 
                               command=self.retrain_model,
                               font=('Segoe UI', 10, 'bold'),
                               bg='#e67e22', fg='white', relief='raised', bd=2,
                               padx=15, pady=8, cursor='hand2')
        retrain_btn.pack(side=tk.LEFT, padx=5)
        retrain_btn.bind('<Enter>', lambda e: self.on_button_hover(e, '#d35400'))
        retrain_btn.bind('<Leave>', lambda e: self.on_button_leave(e, '#e67e22'))
    
    def show_styled_initial_message(self):
        """Show styled initial message in results area"""
        self.results_text.delete(1.0, tk.END)
        
        # Welcome header
        self.results_text.insert(tk.END, "🏥 Heart Disease Risk Assessment System\n\n", 'header')
        
        # Instructions
        self.results_text.insert(tk.END, "Welcome to the Professional Medical Risk Assessment Tool\n\n")
        self.results_text.insert(tk.END, "📋 Instructions:\n", 'info')
        self.results_text.insert(tk.END, "1. Fill in all patient information fields above\n")
        self.results_text.insert(tk.END, "2. Click 'Predict Heart Disease Risk' to get assessment\n")
        self.results_text.insert(tk.END, "3. Review the detailed risk analysis and recommendations\n")
        self.results_text.insert(tk.END, "4. Save predictions for future reference and analysis\n\n")
        
        # Features
        self.results_text.insert(tk.END, "✨ Key Features:\n", 'success')
        self.results_text.insert(tk.END, "• Advanced machine learning risk prediction\n")
        self.results_text.insert(tk.END, "• Comprehensive patient data management\n")
        self.results_text.insert(tk.END, "• Visual risk level interpretation\n")
        self.results_text.insert(tk.END, "• Data export and analysis capabilities\n")
        self.results_text.insert(tk.END, "• Continuous model improvement\n\n")
        
        # Medical disclaimer
        self.results_text.insert(tk.END, "⚠️  IMPORTANT MEDICAL DISCLAIMER:\n", 'warning')
        self.results_text.insert(tk.END, "This tool is designed for educational and research purposes only.\n", 'info')
        self.results_text.insert(tk.END, "Always consult qualified healthcare professionals for medical advice.\n", 'info')
        self.results_text.insert(tk.END, "Do not use this tool as a substitute for professional medical diagnosis.\n", 'info')
        
        self.results_text.config(state=tk.DISABLED)
        
    def load_model_async(self):
        """Load and train model in background"""
        def load_model():
            try:
                self.root.after(0, lambda: self.update_status("📊 Loading dataset...", 'info'))
                
                if not os.path.exists(self.data_file):
                    self.root.after(0, lambda: self.update_status(f"❌ Error: '{self.data_file}' not found!", 'error'))
                    return
                
                # Load data
                heart_data = pd.read_csv(self.data_file)
                X = heart_data.drop(columns='target', axis=1)
                y = heart_data['target']
                
                self.root.after(0, lambda: self.update_status("🤖 Training model...", 'info'))
                
                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                self.model = LogisticRegression(random_state=42, max_iter=1000)
                self.model.fit(X_train, y_train)
                
                # Calculate accuracy
                test_predictions = self.model.predict(X_test)
                accuracy = accuracy_score(y_test, test_predictions)
                
                # Update UI
                self.root.after(0, lambda: self.update_status("✅ Model ready!", 'success'))
                self.root.after(0, lambda: self.model_info_label.config(text=f"Model Accuracy: {accuracy:.1%} | Dataset: {len(heart_data)} records"))
                self.root.after(0, lambda: self.predict_button.config(state='normal'))
                self.root.after(0, self.update_data_info)
                
            except Exception as e:
                self.root.after(0, lambda: self.update_status(f"❌ Error: {str(e)}", 'error'))
        
        threading.Thread(target=load_model, daemon=True).start()
        
    def update_status(self, message, status_type):
        """Update status label"""
        colors = {'info': '#3498db', 'success': '#27ae60', 'error': '#e74c3c'}
        self.status_label.config(text=message, foreground=colors.get(status_type, '#7f8c8d'))
        
    def update_data_info(self):
        """Update data management info"""
        try:
            if os.path.exists(self.user_data_file):
                user_data = pd.read_csv(self.user_data_file)
                self.data_info_label.config(text=f"Saved predictions: {len(user_data)} records")
            else:
                self.data_info_label.config(text="No saved predictions yet")
        except:
            self.data_info_label.config(text="Error reading saved data")
            
    def validate_inputs(self):
        """Validate all inputs"""
        try:
            ranges = {
                'age': (1, 120), 'sex': (0, 1), 'cp': (0, 3), 'trestbps': (50, 300),
                'chol': (100, 600), 'fbs': (0, 1), 'restecg': (0, 2), 'thalach': (50, 250),
                'exang': (0, 1), 'oldpeak': (0, 10), 'slope': (0, 2), 'ca': (0, 4), 'thal': (0, 3)
            }
            
            for feature, (min_val, max_val) in ranges.items():
                value = self.input_vars[feature].get()
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{feature.title()} must be between {min_val} and {max_val}")
            return True
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return False
            
    def predict_risk(self):
        """Make prediction"""
        if not self.model:
            messagebox.showerror("Error", "Model not ready yet")
            return
            
        if not self.validate_inputs():
            return
            
        try:
            # Get input data
            feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            input_data = [self.input_vars[feature].get() for feature in feature_order]
            input_array = np.array(input_data).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(input_array)[0]
            probability = self.model.predict_proba(input_array)[0]
            
            # Store current prediction
            self.current_prediction = {
                'input_data': input_data,
                'prediction': prediction,
                'probability': probability,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Display results
            self.display_results(prediction, probability, input_data)
            self.save_button.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error: {str(e)}")
            
    def display_results(self, prediction, probability, input_data):
        """Display prediction results"""
        self.results_text.delete(1.0, tk.END)
        
        # Header
        self.results_text.insert(tk.END, "=" * 70 + "\n")
        self.results_text.insert(tk.END, "🏥 HEART DISEASE RISK ASSESSMENT RESULTS\n")
        self.results_text.insert(tk.END, "=" * 70 + "\n\n")
        
        # Timestamp
        self.results_text.insert(tk.END, f"📅 Assessment Date: {self.current_prediction['timestamp']}\n\n")
        
        # Main result
        risk_score = probability[1] * 100
        confidence = max(probability) * 100
        
        if prediction == 0:
            self.results_text.insert(tk.END, "✅ RESULT: LOW RISK\n")
            self.results_text.insert(tk.END, f"   The model indicates LOW risk of heart disease\n")
        else:
            self.results_text.insert(tk.END, "⚠️  RESULT: HIGH RISK\n")
            self.results_text.insert(tk.END, f"   The model indicates HIGH risk of heart disease\n")
            
        self.results_text.insert(tk.END, f"   Confidence Level: {confidence:.1f}%\n")
        self.results_text.insert(tk.END, f"   Risk Score: {risk_score:.1f}%\n\n")
        
        # Risk level interpretation
        if risk_score < 25:
            level = "🟢 Very Low Risk"
        elif risk_score < 50:
            level = "🟡 Low Risk"
        elif risk_score < 75:
            level = "🟠 Moderate Risk"
        else:
            level = "🔴 High Risk"
            
        self.results_text.insert(tk.END, f"📊 Risk Level: {level}\n\n")
        
        # Input summary
        self.results_text.insert(tk.END, "📋 Input Summary:\n")
        self.results_text.insert(tk.END, "-" * 30 + "\n")
        
        feature_names = ['Age', 'Sex', 'Chest Pain', 'Resting BP', 'Cholesterol', 'Fasting BS',
                        'Resting ECG', 'Max HR', 'Exercise Angina', 'ST Depression', 'ST Slope',
                        'Major Vessels', 'Thalassemia']
        
        for i, (name, value) in enumerate(zip(feature_names, input_data)):
            self.results_text.insert(tk.END, f"   • {name}: {value}\n")
            
        # Disclaimer
        self.results_text.insert(tk.END, f"\n{'='*70}\n")
        self.results_text.insert(tk.END, "⚠️  MEDICAL DISCLAIMER:\n")
        self.results_text.insert(tk.END, "This prediction is for educational purposes only.\n")
        self.results_text.insert(tk.END, "Always consult healthcare professionals for medical advice.\n")
        self.results_text.insert(tk.END, "=" * 70 + "\n")
        
    def save_prediction(self):
        """Save current prediction to dataset"""
        if not hasattr(self, 'current_prediction'):
            messagebox.showwarning("Warning", "No prediction to save")
            return
            
        try:
            # Prepare data for saving
            feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            new_record = {}
            for i, feature in enumerate(feature_names):
                new_record[feature] = self.current_prediction['input_data'][i]
            
            new_record['target'] = self.current_prediction['prediction']
            new_record['risk_score'] = self.current_prediction['probability'][1] * 100
            new_record['confidence'] = max(self.current_prediction['probability']) * 100
            new_record['timestamp'] = self.current_prediction['timestamp']
            
            # Save to user data file
            if os.path.exists(self.user_data_file):
                user_df = pd.read_csv(self.user_data_file)
                user_df = pd.concat([user_df, pd.DataFrame([new_record])], ignore_index=True)
            else:
                user_df = pd.DataFrame([new_record])
                
            user_df.to_csv(self.user_data_file, index=False)
            
            # Also add to main dataset (optional - ask user)
            result = messagebox.askyesno("Add to Training Data", 
                                       "Would you like to add this record to the main training dataset?")
            if result:
                main_df = pd.read_csv(self.data_file)
                main_record = {feature: new_record[feature] for feature in feature_names}
                main_record['target'] = new_record['target']
                main_df = pd.concat([main_df, pd.DataFrame([main_record])], ignore_index=True)
                main_df.to_csv(self.data_file, index=False)
                
            messagebox.showinfo("Success", "Prediction saved successfully!")
            self.update_data_info()
            self.save_button.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving prediction: {str(e)}")
            
    def clear_fields(self):
        """Clear all input fields"""
        for var in self.input_vars.values():
            var.set(0)
        self.show_initial_message()
        self.save_button.config(state='disabled')
        
    def show_initial_message(self):
        """Show initial message in results area"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "🏥 Heart Disease Risk Assessment System\n\n")
        self.results_text.insert(tk.END, "Enter patient information above and click 'Predict Heart Disease Risk' to get results.\n\n")
        self.results_text.insert(tk.END, "Features:\n")
        self.results_text.insert(tk.END, "• Professional medical risk assessment\n")
        self.results_text.insert(tk.END, "• Save predictions for future reference\n")
        self.results_text.insert(tk.END, "• Export data for analysis\n")
        self.results_text.insert(tk.END, "• Continuous model improvement\n\n")
        self.results_text.insert(tk.END, "⚠️  This tool is for educational purposes only.\n")
        self.results_text.insert(tk.END, "Always consult healthcare professionals for medical advice.")
        
    def view_saved_data(self):
        """View saved predictions"""
        try:
            if not os.path.exists(self.user_data_file):
                messagebox.showinfo("No Data", "No saved predictions found")
                return
                
            user_data = pd.read_csv(self.user_data_file)
            
            # Create new window to display data
            data_window = tk.Toplevel(self.root)
            data_window.title("Saved Predictions")
            data_window.geometry("800x600")
            
            # Create treeview for data display
            tree_frame = ttk.Frame(data_window, padding="10")
            tree_frame.pack(fill=tk.BOTH, expand=True)
            
            tree = ttk.Treeview(tree_frame)
            tree.pack(fill=tk.BOTH, expand=True)
            
            # Configure columns
            tree['columns'] = list(user_data.columns)
            tree['show'] = 'headings'
            
            for col in user_data.columns:
                tree.heading(col, text=col)
                tree.column(col, width=80)
                
            # Insert data
            for _, row in user_data.iterrows():
                tree.insert('', tk.END, values=list(row))
                
        except Exception as e:
            messagebox.showerror("Error", f"Error viewing data: {str(e)}")
            
    def export_data(self):
        """Export saved data"""
        try:
            if not os.path.exists(self.user_data_file):
                messagebox.showinfo("No Data", "No data to export")
                return
                
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                user_data = pd.read_csv(self.user_data_file)
                user_data.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Data exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting data: {str(e)}")
            
    def retrain_model(self):
        """Retrain model with updated data"""
        result = messagebox.askyesno("Retrain Model", 
                                   "This will retrain the model with the latest data. Continue?")
        if result:
            self.predict_button.config(state='disabled')
            self.load_model_async()
            
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main function"""
    app = HeartDiseaseApp()
    app.run()

if __name__ == "__main__":
    main()
