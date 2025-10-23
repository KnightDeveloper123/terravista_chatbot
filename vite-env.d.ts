/// <reference types="vite/client" />

interface ImportMetaEnv {
    readonly VITE_BACKEND_URL: string
    // add other VITE_XXX env variables here
}

interface ImportMeta {
    readonly env: ImportMetaEnv
}
