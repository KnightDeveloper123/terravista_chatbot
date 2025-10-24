import useFetch from '@/hooks/useFetch';
import { formatDate, showAlert } from '@/utils/helpers';
import { Box, Button, CloseButton, Dialog, Field, Fieldset, FileUpload, Flex, GridItem, Heading, Icon, IconButton, Input, Popover, Portal, Separator, SimpleGrid, Skeleton, Table, Text } from '@chakra-ui/react';
import React, { useCallback, useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { LuUpload } from 'react-icons/lu';
import { MdDeleteOutline } from 'react-icons/md';

type DocumentType = {
    id: number
    name: string
    created_at: string
    status: number
}

type FormValues = {
    name: string
    file: FileList
}


const Dashboard: React.FC = () => {
    const { loading, request } = useFetch();
    const [documents, setDocuments] = useState<DocumentType[]>([]);
    const [selectedFile, setSelectedFile] = useState<DocumentType | null>(null);

    const { register, handleSubmit, watch, reset, formState: { errors } } = useForm<FormValues>({
        defaultValues: {
            name: '',
            file: undefined as any,
        },
        mode: 'onTouched',
    })


    const fetchAllDocuments = useCallback(async () => {
        try {
            const response = await request({ url: `/documents/getAllDocuments` });
            if (response.success) {
                setDocuments(response.data)
            } else {
                showAlert(response.error, "", 'error')
            }
        } catch (error) {
            console.log(error)
            showAlert('Internal Server Error', "", 'error')
        }
    }, []);

    function validateFile(files: FileList | undefined) {
        if (!files || files.length === 0) {
            return 'File is required'
        }
        const f = files[0]
        const maxBytes = 5 * 1024 * 1024 // 5 MB
        if (f.size > maxBytes) {
            return 'File must be smaller than 5 MB'
        }
        const allowed = ['.pdf', '.csv', '.txt', '.docx', '.xlsx', '.png', '.jpg', '.jpeg']
        const ok = allowed.some(ext => f.name.toLowerCase().endsWith(ext))
        if (!ok) {
            return 'Unsupported file type'
        }
        return true
    }

    async function onSubmit(data: FormValues) {
        console.log(data);

        const fileList = data.file
        if (!fileList || fileList.length === 0) {
            showAlert('No file', "Please select a valid file", 'error')
            return
        }

        const fd = new FormData()
        fd.append('fileName', data.name)
        fd.append('file', fileList[0])

        try {
            const res = await request({ url: `/documents/uploadDocument`, method: 'POST', body: fd })
            if (res.success) {
                showAlert('Success', res.success, 'success')
                fetchAllDocuments();
            }
            reset();
        } catch (err) {
            console.log(err);
            showAlert('Upload error', err instanceof Error ? err.message : 'Something went wrong', 'error')
        }
    }

    const deleteDocument = useCallback(async (id: number) => {
        try {
            const response = await request({ url: `/documents/deleteDocument?document_id=${id}` });
            if (response.success) {
                fetchAllDocuments();
                showAlert("Deleted Succesfully", "", 'success')
            } else {
                showAlert(response.error, "", 'error')
            }
        } catch (error) {
            console.log(error)
            showAlert('Internal Server Error', "", 'error')
        }
    }, []);

    useEffect(() => {
        fetchAllDocuments();
    }, [])

    return (
        <Box p={2} bg={'#fff'} borderRadius={'10px'}>
            <Flex alignItems="center" justifyContent={'space-between'}>
                <Heading>Documents</Heading>

                <Dialog.Root motionPreset="slide-in-bottom">
                    <Dialog.Trigger asChild>
                        <Button colorPalette={'blue'} variant={'surface'} size={'xs'}>Add Document</Button>
                    </Dialog.Trigger>
                    <Portal>
                        <Dialog.Backdrop />
                        <Dialog.Positioner>
                            <Dialog.Content>
                                <Dialog.Header>
                                    <Dialog.Title>Add Document</Dialog.Title>
                                </Dialog.Header>
                                <Dialog.Body>
                                    <Box as={'form'} onSubmit={handleSubmit(onSubmit)}>
                                        <Fieldset.Root size="lg" invalid>
                                            <Fieldset.Content>
                                                <Field.Root invalid={!!errors.name}>
                                                    <Field.Label>File Name</Field.Label>
                                                    <Input placeholder='File Name' {...register('name', {
                                                        required: 'File name is required',
                                                        minLength: { value: 2, message: 'Too short' },
                                                    })} />
                                                    <Field.ErrorText>{errors.name && errors.name.message}</Field.ErrorText>
                                                </Field.Root>
                                                <FileUpload.Root maxW="xl" alignItems="stretch" maxFiles={10} invalid={!!errors.file}>
                                                    <FileUpload.HiddenInput
                                                        accept=".pdf,.csv,.txt,.docx,.xlsx,.png,.jpg,.jpeg"
                                                        {...register('file', {
                                                            validate: validateFile,
                                                            required: true,
                                                        })} />

                                                    <FileUpload.Dropzone>
                                                        <Icon size="md" color="fg.muted">
                                                            <LuUpload />
                                                        </Icon>
                                                        <FileUpload.DropzoneContent>
                                                            <Box>Drag and drop files here</Box>
                                                            <Box color="fg.muted">.png, .jpg up to 5MB</Box>
                                                        </FileUpload.DropzoneContent>
                                                    </FileUpload.Dropzone>
                                                    <FileUpload.List />
                                                </FileUpload.Root>
                                            </Fieldset.Content>
                                        </Fieldset.Root>


                                        <Flex justifyContent={'flex-end'} mt={6} gap={2}>
                                            <Dialog.ActionTrigger asChild>
                                                <Button size={'xs'} variant="outline">Cancel</Button>
                                            </Dialog.ActionTrigger>
                                            <Button size={'xs'} type='submit'>Save</Button>
                                        </Flex>
                                    </Box>
                                </Dialog.Body>
                                <Dialog.CloseTrigger asChild>
                                    <CloseButton size="sm" />
                                </Dialog.CloseTrigger>
                            </Dialog.Content>
                        </Dialog.Positioner>
                    </Portal>
                </Dialog.Root>

            </Flex>
            <Separator my={2} />
            {loading ? <Skeleton h={'350px'} /> : <SimpleGrid columns={{ base: 1, md: 2 }} gap={2}>
                <GridItem>
                    <Table.ScrollArea mt={2} borderWidth="1px" rounded="md">
                        <Table.Root size="sm" stickyHeader>
                            <Table.Caption bg="bg.subtle" captionSide="top">
                                DATA SETS UPLOADED FOR MODEL TRAINING
                            </Table.Caption>
                            <Table.Header>
                                <Table.Row bg="bg.subtle">
                                    <Table.ColumnHeader>Product</Table.ColumnHeader>
                                    <Table.ColumnHeader>Date</Table.ColumnHeader>
                                    <Table.ColumnHeader textAlign="end">Action</Table.ColumnHeader>
                                </Table.Row>
                            </Table.Header>

                            <Table.Body>
                                {documents.map((item) => (
                                    <Table.Row key={item.id} fontSize={'13px'} _hover={{ bg: "#f3f3f3ff", cursor: 'pointer' }} onClick={() => setSelectedFile(item)}>
                                        <Table.Cell>{item.name}</Table.Cell>
                                        <Table.Cell>{formatDate(item.created_at)}</Table.Cell>
                                        <Table.Cell textAlign="end" onClick={(e) => e.stopPropagation()}>
                                            <Popover.Root>
                                                <Popover.Trigger asChild>
                                                    <IconButton size="sm" variant="ghost" colorPalette={'red'}><MdDeleteOutline /></IconButton>
                                                </Popover.Trigger>
                                                <Portal>
                                                    <Popover.Positioner>
                                                        <Popover.Content>
                                                            <Popover.Arrow />
                                                            <Popover.Body>
                                                                <Text fontSize={'13px'}>Are you sure you want to delete the file.</Text>
                                                                <Button colorPalette={'red'} size={'xs'} mt={4} float={'right'} onClick={() => deleteDocument(item.id)}>Delete</Button>
                                                            </Popover.Body>
                                                        </Popover.Content>
                                                    </Popover.Positioner>
                                                </Portal>
                                            </Popover.Root>
                                        </Table.Cell>
                                    </Table.Row>
                                ))}
                            </Table.Body>
                        </Table.Root>
                    </Table.ScrollArea>
                </GridItem>
                <GridItem p={2}>
                    {selectedFile ? <FileViewer selectedFile={selectedFile} /> : <Text fontSize={'13px'}>No File Selected</Text>}
                </GridItem>
            </SimpleGrid>}
        </Box>
    );
};



const FileViewer = ({ selectedFile }: { selectedFile: DocumentType }) => {
    const [fileUrl, setFileUrl] = useState('')
    const [fileContent, setFileContent] = useState('')

    useEffect(() => {
        if (!selectedFile) {
            setFileUrl('')
            setFileContent('')
            return
        }

        const filePath = `${import.meta.env.VITE_BACKEND_URL}/documents/${selectedFile.name}`

        if (selectedFile.name.endsWith('.pdf')) {
            // Just set URL for iframe
            setFileUrl(filePath)
        } else if (
            selectedFile.name.endsWith('.csv') ||
            selectedFile.name.endsWith('.txt')
        ) {
            fetch(filePath)
                .then(res => (res.ok ? res.text() : Promise.reject('Failed to fetch')))
                .then(setFileContent)
                .catch(() => setFileContent('Error loading file.'))
        } else {
            setFileUrl('')
            setFileContent('')
        }
    }, [selectedFile])

    return (
        <GridItem borderRadius='md' minH='300px' maxH='500px' overflowY='auto'>
            {selectedFile ? (
                <>
                    <Text fontWeight='bold' mb={2}>
                        {selectedFile.name}
                    </Text>

                    {selectedFile.name.endsWith('.pdf') ? (
                        <iframe
                            src={fileUrl}
                            style={{
                                width: '100%',
                                height: '400px',
                                border: 'none',
                                borderRadius: '8px',
                            }}
                            title='PDF Viewer'
                        />
                    ) : ['.csv', '.txt'].some(ext => selectedFile.name.endsWith(ext)) ? (
                        <Box
                            as='pre'
                            whiteSpace='pre-wrap'
                            wordBreak='break-word'
                            fontSize='sm'
                            bg='gray.50'
                            p={3}
                            borderRadius='md'
                            overflowX='auto'
                        >
                            {fileContent || 'Loading file content...'}
                        </Box>
                    ) : (
                        <Text>Preview not available for this file type.</Text>
                    )}
                </>
            ) : (
                <Text>Select a file to view its content.</Text>
            )}
        </GridItem>
    )
}



export default Dashboard;