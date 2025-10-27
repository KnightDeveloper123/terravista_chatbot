import useFetch from '@/hooks/useFetch';
import { formatDate, showAlert } from '@/utils/helpers';
import { Box, Heading, IconButton, Menu, Portal, Table } from '@chakra-ui/react';
import React, { useCallback, useEffect, useState } from 'react';
import { CiSettings } from 'react-icons/ci';

type User = {
    id: number
    name: string
    email: string
    mobile_no: string
    status: 0 | 1
    created_at: string
    updated_at: string
    last_login: string
    date_of_birth: string | null
    account_status: 0 | 1
}


const Users: React.FC = () => {
    const { request, loading } = useFetch();

    const [users, setUsers] = useState<User[]>([]);
    const fetchAllUsers = useCallback(async () => {
        try {
            const response = await request({ url: `/user/getAllUser` });
            if (response.success) {
                setUsers(response.data)
            } else {
                showAlert(response.error, "", 'error')
            }
        } catch (error) {
            console.log(error)
            showAlert('Internal Server Error', "", 'error')
        }
    }, []);

    useEffect(() => {
        fetchAllUsers();
    }, []);

    const deleteUser = useCallback(async (id: number) => {
        try {
            const response = await request({ url: `/user/deleteUser?user_id=${id}` });
            if (response.success) {
                fetchAllUsers();
                showAlert("Deleted Succesfully", "", 'success')
            } else {
                showAlert(response.error, "", 'error')
            }
        } catch (error) {
            console.log(error)
            showAlert('Internal Server Error', "", 'error')
        }
    }, []);
    return (
        <Box p={2} bg={'#fff'} borderRadius={'10px'}>

            <Heading size="md" mb={4}>Users</Heading>

            <Table.ScrollArea mt={2} borderWidth="1px" rounded="md">
                <Table.Root size="sm" stickyHeader>
                    <Table.Header>
                        <Table.Row bg="bg.subtle">
                            <Table.ColumnHeader>id</Table.ColumnHeader>
                            <Table.ColumnHeader>name</Table.ColumnHeader>
                            <Table.ColumnHeader>email</Table.ColumnHeader>
                            <Table.ColumnHeader>mobile_no</Table.ColumnHeader>
                            <Table.ColumnHeader>DOB</Table.ColumnHeader>
                            <Table.ColumnHeader>Last Login</Table.ColumnHeader>
                            <Table.ColumnHeader>Created At</Table.ColumnHeader>
                            <Table.ColumnHeader textAlign="end">Action</Table.ColumnHeader>
                        </Table.Row>
                    </Table.Header>

                    <Table.Body>
                        {users.map((item) => (
                            <Table.Row key={item.id} fontSize={'13px'} _hover={{ bg: "#f3f3f3ff", cursor: 'pointer' }}>
                                <Table.Cell>{item.id}</Table.Cell>
                                <Table.Cell>{item.name}</Table.Cell>
                                <Table.Cell>{item.email}</Table.Cell>
                                <Table.Cell>{item.mobile_no}</Table.Cell>
                                <Table.Cell>{item.date_of_birth}</Table.Cell>
                                <Table.Cell>{formatDate(item.last_login)}</Table.Cell>
                                <Table.Cell>{formatDate(item.created_at)}</Table.Cell>
                                <Table.Cell textAlign="end" onClick={(e) => e.stopPropagation()}>
                                    <Menu.Root>
                                        <Menu.Trigger asChild>
                                            <IconButton size="sm" variant="ghost" colorPalette={'red'}><CiSettings /></IconButton>
                                        </Menu.Trigger>
                                        <Portal>
                                            <Menu.Positioner>
                                                <Menu.Content>
                                                    <Menu.Item value="account-status">Account Status</Menu.Item>
                                                    <Menu.Item color="fg.error" _hover={{ bg: "bg.error", color: "fg.error" }} onClick={() => deleteUser(item.id)} value="delete">Delete...</Menu.Item>
                                                </Menu.Content>
                                            </Menu.Positioner>
                                        </Portal>
                                    </Menu.Root>
                                </Table.Cell>
                            </Table.Row>
                        ))}
                    </Table.Body>
                </Table.Root>
            </Table.ScrollArea>
        </Box>
    );
};

export default Users;