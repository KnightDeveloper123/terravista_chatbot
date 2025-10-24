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

export const formatDate = (isoString: string): string => {
    const date = new Date(isoString)
    return date.toLocaleString('en-GB', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true,
    })
}