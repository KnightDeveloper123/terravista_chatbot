import { toaster } from "@/components/ui/toaster"

type AlertType = "success" | "error" | "info" | "warning";


export const showAlert = (title: string, description: string = "", type: AlertType) => {
    return toaster.create({
        title,
        description,
        type: type,
        duration: 2000
    })
}