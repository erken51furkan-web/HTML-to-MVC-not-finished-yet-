import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import shutil
import threading
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Arayüz yapılandırması
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HTML'den MVC'ye Dönüştürücü (AI Destekli)")
        self.geometry("700x550")
        self.resizable(False, False)

        self.source_path = ""
        self.target_path = ""
        self.model_path = "layout_model.pkl"

        self._build_ui()

    def _build_ui(self):
        self.lbl_title = ctk.CTkLabel(self, text="Otomatik Şablon Dönüştürme Motoru", font=ctk.CTkFont(size=20, weight="bold"))
        self.lbl_title.pack(pady=(15, 5))

        # --- YAPAY ZEKA EĞİTİM MODÜLÜ ---
        self.frame_ai = ctk.CTkFrame(self)
        self.frame_ai.pack(pady=5, padx=20, fill="x")
        self.btn_train = ctk.CTkButton(self.frame_ai, text="1. Yapay Zeka Modelini Eğit", fg_color="#8e44ad", hover_color="#9b59b6", command=self.start_training)
        self.btn_train.pack(side="left", padx=10, pady=10)
        self.lbl_ai_status = ctk.CTkLabel(self.frame_ai, text="Model durumu kontrol ediliyor...", text_color="gray")
        self.lbl_ai_status.pack(side="left", padx=10)

        # --- KAYNAK VE HEDEF SEÇİMİ ---
        self.frame_source = ctk.CTkFrame(self)
        self.frame_source.pack(pady=5, padx=20, fill="x")
        self.btn_source = ctk.CTkButton(self.frame_source, text="Kaynak HTML Klasörü Seç", command=self.select_source)
        self.btn_source.pack(side="left", padx=10, pady=10)
        self.lbl_source = ctk.CTkLabel(self.frame_source, text="Henüz bir kaynak seçilmedi...", text_color="gray")
        self.lbl_source.pack(side="left", padx=10)

        self.frame_target = ctk.CTkFrame(self)
        self.frame_target.pack(pady=5, padx=20, fill="x")
        self.btn_target = ctk.CTkButton(self.frame_target, text="Hedef MVC Projesi Seç", command=self.select_target)
        self.btn_target.pack(side="left", padx=10, pady=10)
        self.lbl_target = ctk.CTkLabel(self.frame_target, text="Henüz bir hedef seçilmedi...", text_color="gray")
        self.lbl_target.pack(side="left", padx=10)

        # --- DURUM VE ANA TETİKLEYİCİ ---
        self.lbl_status = ctk.CTkLabel(self, text="Sistem hazır. Lütfen klasör dizinlerini belirleyin.", font=ctk.CTkFont(size=14))
        self.lbl_status.pack(pady=(10, 5))

        self.btn_convert = ctk.CTkButton(self, text="2. Dönüştürmeyi Başlat", font=ctk.CTkFont(size=16, weight="bold"), height=40, command=self.start_conversion, state="disabled")
        self.btn_convert.pack(pady=5, padx=20, fill="x")

        self.check_model_exists()

    def check_model_exists(self):
        if os.path.exists(self.model_path):
            self.lbl_ai_status.configure(text="Model hazır (layout_model.pkl).", text_color="#2ecc71")
        else:
            self.lbl_ai_status.configure(text="Model eksik! Lütfen eğitin.", text_color="#e74c3c")

    # ====================================================================
    # YAPAY ZEKA EĞİTİM MOTORU
    # ====================================================================
    def start_training(self):
        self.btn_train.configure(state="disabled")
        self.lbl_ai_status.configure(text="Eğitim yapılıyor...", text_color="white")
        threading.Thread(target=self.train_and_save_model, daemon=True).start()

    def train_and_save_model(self):
        dataset_path = "dataset.csv"
        if not os.path.exists(dataset_path):
            self.lbl_ai_status.configure(text="Hata: dataset.csv bulunamadı!", text_color="#e74c3c")
            self.btn_train.configure(state="normal")
            return
        try:
            df = pd.read_csv(dataset_path)
            # NLP özelliklerini okuyacak şekilde güncellendi
            X = df[['Tag_Type', 'Text_Density', 'Child_Count', 'Has_Id', 'Has_Template_Keyword', 'Has_Content_Keyword']]
            y = df['IsMainContent']
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            joblib.dump(model, self.model_path)
            self.lbl_ai_status.configure(text="Yapay zeka modeli başarıyla eğitildi.", text_color="#2ecc71")
        except Exception as e:
            self.lbl_ai_status.configure(text=f"Eğitim hatası: {str(e)}", text_color="#e74c3c")
        finally:
            self.btn_train.configure(state="normal")

    # ====================================================================
    # ARAYÜZ KONTROL VE DOSYA İŞLEMLERİ
    # ====================================================================
    def select_source(self):
        path = filedialog.askdirectory(title="Dönüştürülecek HTML Şablon Klasörünü Seçin")
        if path:
            self.source_path = path
            self.lbl_source.configure(text=self.source_path, text_color="white")
            self.validate_ready_state()

    def select_target(self):
        path = filedialog.askdirectory(title="Hedef ASP.NET Core MVC Projesini Seçin")
        if path:
            self.target_path = path
            self.lbl_target.configure(text=self.target_path, text_color="white")
            self.validate_ready_state()

    def validate_ready_state(self):
        if self.source_path and self.target_path:
            self.btn_convert.configure(state="normal")
            self.lbl_status.configure(text="Klasörler doğrulandı. Dönüştürmeyi başlatabilirsiniz.", text_color="#2ecc71")

    def update_status(self, text, color="white"):
        self.lbl_status.configure(text=text, text_color=color)

    def start_conversion(self):
        if not os.path.exists(self.model_path):
            messagebox.showwarning("Uyarı", "Yapay zeka modeli bulunamadı! Önce 1. Butona basarak modeli eğitin.")
            return
        self.btn_convert.configure(state="disabled")
        threading.Thread(target=self.run_pipeline, daemon=True).start()

    # ====================================================================
    # DÖNÜŞTÜRME MOTORU VE AI TAHMİNİ (NLP EKLENTİLİ)
    # ====================================================================
    def run_pipeline(self):
        try:
            self.update_status("Adım 1: Statik dosyalar wwwroot dizinine kopyalanıyor...")
            self.copy_static_files()

            self.update_status("Adım 2: HTML dosyaları işleniyor ve AI ile Layout parçalanıyor...")
            self.process_html_files()

            self.update_status("İşlem başarıyla tamamlandı! MVC projeniz hazır.", "#2ecc71")
        except Exception as e:
            self.update_status(f"Hata oluştu: {str(e)}", "#e74c3c")
            messagebox.showerror("Kritik Hata", f"İşlem sırasında bir hata meydana geldi:\n\n{str(e)}")
        finally:
            self.btn_convert.configure(state="normal")

    def copy_static_files(self):
        wwwroot_path = os.path.join(self.target_path, "wwwroot")
        os.makedirs(wwwroot_path, exist_ok=True)
        static_folders = ["css", "js", "img", "images", "fonts", "assets", "vendor", "scss"]
        for folder in static_folders:
            source_dir = os.path.join(self.source_path, folder)
            target_dir = os.path.join(wwwroot_path, folder)
            if os.path.exists(source_dir) and os.path.isdir(source_dir):
                shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    def is_local_path(self, path):
        if not path or path.lower().startswith(("http", "//", "~", "data:")):
            return False
        return True
    def process_html_files(self):
        model = joblib.load(self.model_path)
        tag_map = {'div': 1, 'section': 2, 'main': 3, 'article': 4, 'header': 5, 'footer': 6, 'nav': 7, 'aside': 8}
        
        template_keywords = ['footer', 'header', 'nav', 'menu', 'sidebar', 'widget', 'topbar', 'bottom']
        content_keywords = ['main', 'content', 'article', 'post']
        wrapper_keywords = ['page', 'wrapper', 'container', 'wrap', 'layout']

        html_files = [f for f in os.listdir(self.source_path) if f.endswith(".html")]

        # YENİ: Dosyaları karakter yoğunluğuna göre büyükten küçüğe sırala.
        # Böylece AI, en net kararı verebileceği sayfayı (en dolu sayfayı) İLK işler!
        html_files.sort(key=lambda x: os.path.getsize(os.path.join(self.source_path, x)), reverse=True)

        # YENİ: SİSTEM HAFIZASI - İlk başarılı tespitte hedefin imzasını buraya kaydedeceğiz
        layout_signature_id = None
        layout_signature_class = None

        for file_name in html_files:
            file_path = os.path.join(self.source_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file, "html.parser")

            # 1. Yolları (Paths) Düzelt
            for link in soup.find_all("link", rel="stylesheet"):
                if self.is_local_path(link.get("href")): link["href"] = f"~/{link.get('href')}"
            for script in soup.find_all("script", src=True):
                if self.is_local_path(script.get("src")): script["src"] = f"~/{script.get('src')}"
            for img in soup.find_all("img", src=True):
                if self.is_local_path(img.get("src")): img["src"] = f"~/{img.get('src')}"

            body_element = soup.find("body")

            # 2. AI İle Ana İçerik Tespiti
            candidates = soup.find_all(['div', 'section', 'main', 'article'])
            best_candidate = None
            ai_approved_candidates = []
            
            for element in candidates:
                tag_type = tag_map.get(element.name, 0)
                text_density = len(element.get_text(strip=True))
                child_count = len(element.find_all(recursive=False))
                has_id = 1 if element.get('id') else 0
                
                element_id = element.get('id', '').lower() if element.get('id') else ''
                element_classes = ' '.join(element.get('class', [])).lower() if element.get('class') else ''
                combined_attributes = f"{element_id} {element_classes}"

                has_wrapper_kw = any(kw in combined_attributes for kw in wrapper_keywords)
                if has_wrapper_kw :
                    continue 

                has_template_kw = 1 if any(kw in combined_attributes for kw in template_keywords) else 0
                has_content_kw = 1 if any(kw in combined_attributes for kw in content_keywords) else 0

                input_data = pd.DataFrame([[tag_type, text_density, child_count, has_id, has_template_kw, has_content_kw]], 
                                          columns=['Tag_Type', 'Text_Density', 'Child_Count', 'Has_Id', 'Has_Template_Keyword', 'Has_Content_Keyword'])
                
                prediction = model.predict(input_data)[0]
                
                if prediction == 1 and text_density > 50:
                    ai_approved_candidates.append((element, text_density))

            # Karar Aşaması
            if ai_approved_candidates:
                ai_approved_candidates.sort(key=lambda x: x[1], reverse=True)
                best_candidate = ai_approved_candidates[0][0]

                # YENİ: GELECEK ZAYIF SAYFALAR İÇİN HEDEFIN İMZASINI HAFIZAYA AL
                if best_candidate.get('id'):
                    layout_signature_id = best_candidate.get('id')
                elif best_candidate.get('class'):
                    layout_signature_class = best_candidate.get('class')[0] # İlk class adını al
            
            # YENİ: EĞER AI BU SAYFADA ÇUVALLADIYSA (Örn: Sadece formu olan Contact sayfası)
            if not best_candidate:
                if layout_signature_id:
                    best_candidate = soup.find(id=layout_signature_id) # Hafızadaki ID'ye göre bul
                elif layout_signature_class:
                    best_candidate = soup.find(class_=layout_signature_class) # Hafızadaki Class'a göre bul

            # 3. Şablonu Ayrıştır
            if best_candidate:
                view_content = best_candidate.decode_contents()
                best_candidate.clear()
                best_candidate.append("___RENDER_BODY_PLACEHOLDER___")
            else:
                if body_element:
                    view_content = body_element.decode_contents()
                    body_element.clear()
                    body_element.append("___RENDER_BODY_PLACEHOLDER___")
                else:
                    view_content = "<!-- İçerik bulunamadı -->"

            layout_html = str(soup).replace("___RENDER_BODY_PLACEHOLDER___", "\n    @RenderBody()\n")
            self.save_mvc_files(layout_html, view_content, file_name) 
            
    def save_mvc_files(self, layout_html, view_html, original_filename):
        views_dir = os.path.join(self.target_path, "Views")
        shared_dir = os.path.join(views_dir, "Shared")
        home_dir = os.path.join(views_dir, "Home")

        os.makedirs(shared_dir, exist_ok=True)
        os.makedirs(home_dir, exist_ok=True)

        layout_path = os.path.join(shared_dir, "_Layout.cshtml")
        if not os.path.exists(layout_path):
            with open(layout_path, "w", encoding="utf-8") as f:
                f.write(layout_html)

        view_name = os.path.splitext(original_filename)[0].capitalize() + ".cshtml"
        view_path = os.path.join(home_dir, view_name)
        
        with open(view_path, "w", encoding="utf-8") as f:
            f.write(f"@{{\n    Layout = \"_Layout\";\n}}\n\n{view_html}")

if __name__ == "__main__":
    app = App()
    app.mainloop()