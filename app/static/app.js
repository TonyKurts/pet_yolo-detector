document.addEventListener("alpine:init", () => {
    Alpine.data("visitorCounter", () => ({
        history: [],
        selectedFile: null,
        busy: false,
        loading: true,
        error: "",
        taskId: null,
        pollTimer: null,

        get canSubmit() {
            return Boolean(this.selectedFile) && !this.busy;
        },

        async init() {
            await Promise.all([this.refreshHistory(), this.checkInitialStatus()]);
            this.loading = false;
        },

        async checkInitialStatus() {
            try {
                const data = await this.fetchJson("/api/status");
                this.busy = !data.available;

                if (this.busy && data.task) {
                    this.taskId = data.task.id;
                    this.schedulePoll();
                }
            } catch (error) {
                this.error = error.message;
            }
        },

        selectFile(file) {
            this.error = "";

            if (!file) {
                this.selectedFile = null;
                return;
            }

            const extension = file.name.split(".").pop().toLowerCase();
            
            if (!["png", "jpg", "jpeg"].includes(extension)) {
                this.selectedFile = null;
                this.$refs.fileInput.value = "";
                this.error = "Выберите файл в формате PNG, JPG или JPEG.";
                return;
            }

            this.selectedFile = file;
        },

        handleDrop(event) {
            if (!this.busy) this.selectFile(event.dataTransfer.files[0]);
        },

        async upload() {
            if (!this.canSubmit) return;
            
            this.error = "";
            this.busy = true;

            const form = new FormData();
            form.append("file", this.selectedFile);
            
            try {
                const response = await fetch("/api/upload", { method: "POST", body: form });
                const data = await response.json();
                
                if (!response.ok) throw new Error(data.detail || "Не удалось отправить изображение.");
                
                this.taskId = data.task_id;
                this.selectedFile = null;
                this.$refs.fileInput.value = "";
                
                await this.refreshHistory();
                this.schedulePoll();
            } catch (error) {
                this.busy = false;
                this.error = error.message;
                await this.refreshHistory();
            }
        },

        schedulePoll() {
            window.clearTimeout(this.pollTimer);
            this.pollTimer = window.setTimeout(() => this.pollStatus(), 2000);
        },

        async pollStatus() {
            try {
                const query = this.taskId ? `?task_id=${encodeURIComponent(this.taskId)}` : "";
                const data = await this.fetchJson(`/api/status${query}`);
                
                this.busy = !data.available;
               
                if (this.busy) {
                    this.schedulePoll();
                    return;
                }
                
                this.taskId = null;
                await this.refreshHistory();
            } catch (error) {
                this.error = error.message;
                this.schedulePoll();
            }
        },

        async refreshHistory() {
            try {
                this.history = await this.fetchJson("/api/history");
            } catch (error) {
                this.error = error.message;
            }
        },

        async fetchJson(url) {
            const response = await fetch(url);
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Ошибка связи с сервером.");
            return data;
        },

        statusLabel(status) {
            return { 0: "Новая", 1: "В процессе", 2: "Готово", 3: "Ошибка" }[status] || "Неизвестно";
        },

        formatDate(value) {
            return new Intl.DateTimeFormat("ru-RU", {
                day: "2-digit",
                month: "2-digit",
                year: "numeric",
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
            }).format(new Date(value));
        },

        formatBytes(bytes) {
            return bytes < 1024 * 1024 ? `${Math.ceil(bytes / 1024)} КБ` : `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
        },

        storageUrl(path) {
            return `/storage/${path.split("/").map(encodeURIComponent).join("/")}`;
        },
    }));
});
